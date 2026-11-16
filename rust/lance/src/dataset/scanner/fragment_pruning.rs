// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::dataset::overlay::{collect_overlay_stale_frags, overlaid_fragments};
use crate::index::DatasetIndexInternalExt;
use crate::index::scalar_logical::load_named_scalar_segments;
use async_recursion::async_recursion;
use datafusion::common::ScalarValue;
use lance_index::metrics::{MetricsCollector, NoOpMetricsCollector};
use lance_index::scalar::expression::{ScalarIndexExpr, ScalarIndexSearch};

/// Disjoint conservative candidates. Both sets require the full filter on read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentCandidates {
    /// Candidates retained by usable indices; matches are not guaranteed.
    pub hits: RoaringBitmap,
    /// Candidates with missing coverage or unsupported index capabilities.
    pub unknown: RoaringBitmap,
}
impl FragmentCandidates {
    /// All fragments that may contain matching rows.
    pub fn to_scan(&self) -> RoaringBitmap {
        &self.hits | &self.unknown
    }
}
#[derive(Debug)]
struct Candidates {
    possible: RoaringBitmap,
    unresolved: RoaringBitmap,
}

impl Candidates {
    fn unknown(scope: &RoaringBitmap) -> Self {
        Self {
            possible: scope.clone(),
            unresolved: scope.clone(),
        }
    }

    fn and(self, other: Self) -> Self {
        let possible = self.possible & other.possible;
        let unresolved = (&self.unresolved | &other.unresolved) & &possible;
        Self {
            possible,
            unresolved,
        }
    }

    fn or(self, other: Self) -> Self {
        Self {
            possible: self.possible | other.possible,
            unresolved: self.unresolved | other.unresolved,
        }
    }
}

impl Scanner {
    /// Return conservative scalar-filter candidates using only indices and metadata.
    /// Projection, ordering, limit and offset do not affect this result.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let mut scanner = dataset.scan();
    /// scanner.filter("id = 42")?;
    /// let fragments = scanner.candidate_fragments().await?.to_scan();
    /// # Ok(())
    /// # }
    /// ```
    pub async fn candidate_fragments(&self) -> Result<FragmentCandidates> {
        if self.nearest.is_some()
            || self.full_text_query.is_some()
            || self.minhash_query.is_some()
            || self.filter.query_filter.is_some()
        {
            return Err(Error::not_supported(
                "candidate fragments require a scalar filter without vector, full-text, MinHash or query filters",
            ));
        }
        let fragments = self
            .fragments
            .as_deref()
            .unwrap_or(self.dataset.fragments());
        let scope: RoaringBitmap = fragments
            .iter()
            .filter(|f| f.num_rows() != Some(0))
            .map(|f| {
                u32::try_from(f.id)
                    .map_err(|_| Error::invalid_input(format!("Fragment ID {} exceeds u32", f.id)))
            })
            .collect::<Result<_>>()?;
        let plan = self.create_filter_plan(true, None, None).await?;
        let evaluator = Evaluator {
            dataset: &self.dataset,
            scope,
        };
        if matches!(
            plan.expr_filter_plan.full_expr.as_ref(),
            Some(Expr::Literal(
                ScalarValue::Boolean(Some(false)) | ScalarValue::Boolean(None) | ScalarValue::Null,
                _
            ))
        ) {
            return Ok(FragmentCandidates {
                hits: RoaringBitmap::new(),
                unknown: RoaringBitmap::new(),
            });
        }
        let result = match plan.expr_filter_plan.index_query.as_ref() {
            Some(query) => {
                evaluator
                    .evaluate_index(query, &NoOpMetricsCollector)
                    .await?
            }
            None => Candidates::unknown(&evaluator.scope),
        };
        Ok(FragmentCandidates {
            hits: &result.possible - &result.unresolved,
            unknown: result.unresolved,
        })
    }
}
struct Evaluator<'a> {
    dataset: &'a Dataset,
    scope: RoaringBitmap,
}
impl Evaluator<'_> {
    #[async_recursion]
    async fn evaluate_index(
        &self,
        expr: &ScalarIndexExpr,
        metrics: &dyn MetricsCollector,
    ) -> Result<Candidates> {
        match expr {
            ScalarIndexExpr::And(left, right) => {
                let left = self.evaluate_index(left, metrics).await?;
                if left.possible.is_empty() {
                    return Ok(left);
                }
                Ok(left.and(self.evaluate_index(right, metrics).await?))
            }
            ScalarIndexExpr::Or(left, right) => Ok(self
                .evaluate_index(left, metrics)
                .await?
                .or(self.evaluate_index(right, metrics).await?)),
            // Existence in a fragment cannot be complemented: both a predicate
            // and its negation can match different rows in the same fragment.
            ScalarIndexExpr::Not(_) => Ok(Candidates::unknown(&self.scope)),
            ScalarIndexExpr::Query(search) => self.search(search, metrics).await,
        }
    }

    async fn search(
        &self,
        search: &ScalarIndexSearch,
        metrics: &dyn MetricsCollector,
    ) -> Result<Candidates> {
        let scope = &self.scope;
        // The logical loader filters out unknown coverage. Inspect it first:
        // absence of a bitmap must not become evidence of an empty segment.
        if crate::index::loading::load_all_indices(self.dataset)
            .await?
            .iter()
            .any(|segment| segment.name == search.index_name && segment.fragment_bitmap.is_none())
        {
            return Ok(Candidates::unknown(scope));
        }
        let segments =
            load_named_scalar_segments(self.dataset, &search.column, &search.index_name).await?;
        let overlaid = overlaid_fragments(self.dataset.fragments());
        let mut covered = RoaringBitmap::new();
        let mut hits = RoaringBitmap::new();
        let mut unresolved = RoaringBitmap::new();
        for segment in segments {
            let Some(coverage) = &segment.fragment_bitmap else {
                // A segment with unknown coverage might contribute any match.
                return Ok(Candidates::unknown(scope));
            };
            let mut coverage = coverage & scope;
            if coverage.is_empty() {
                continue;
            }
            let mut stale = RoaringBitmap::new();
            collect_overlay_stale_frags(&segment, &overlaid, &mut stale, self.dataset.schema())?;
            stale &= &coverage;
            unresolved |= &stale;
            coverage -= stale;
            if coverage.is_empty() {
                continue;
            }
            let index = self
                .dataset
                .open_scalar_index(&search.column, &segment.uuid, metrics)
                .await?;
            match index
                .search_fragments(
                    search.query.as_ref(),
                    self.dataset.manifest.uses_stable_row_ids(),
                    metrics,
                )
                .await?
            {
                Some(fragments) => {
                    hits |= fragments & &coverage;
                    covered |= coverage;
                }
                None => unresolved |= coverage,
            }
        }
        unresolved |= scope - covered;
        Ok(Candidates {
            possible: hits | &unresolved,
            unresolved,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{RecordBatch, RecordBatchIterator, UInt64Array, record_batch};
    use lance_core::ROW_ADDR;
    use lance_core::utils::address::RowAddress;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_index::IndexType;
    use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
    use lance_io::object_store::ObjectStoreParams;
    use rstest::rstest;

    use crate::Dataset;
    use crate::dataset::builder::DatasetBuilder;
    use crate::dataset::{ReadParams, WriteParams};
    use crate::index::{CreateIndexBuilder, DatasetIndexExt};
    use crate::utils::test::FailingProxyStore;

    fn reader(
        batch: RecordBatch,
    ) -> RecordBatchIterator<
        impl Iterator<Item = std::result::Result<RecordBatch, arrow_schema::ArrowError>>,
    > {
        let schema = batch.schema();
        RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema)
    }

    async fn query(dataset: &Dataset, filter: &str) -> FragmentCandidates {
        let mut scanner = dataset.scan();
        scanner.filter(filter).unwrap();
        scanner.candidate_fragments().await.unwrap()
    }

    async fn reopen_without_data(uri: &str) -> Dataset {
        let proxy = Arc::new(FailingProxyStore::new());
        for method in ["get_opts", "get_range", "get_ranges", "head"] {
            proxy.fail_when(method, "/data/", "fragment pruning attempted a data read");
            proxy.fail_when(
                method,
                "/_deletions/",
                "fragment pruning attempted a deletion read",
            );
        }
        DatasetBuilder::from_uri(uri)
            .with_read_params(ReadParams {
                store_options: Some(ObjectStoreParams {
                    object_store_wrapper: Some(proxy),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .load()
            .await
            .unwrap()
    }

    #[rstest]
    #[tokio::test]
    async fn test_index_only_candidates(
        #[values(BuiltinIndexType::BTree, BuiltinIndexType::ZoneMap)] kind: BuiltinIndexType,
        #[values(false, true)] stable: bool,
    ) {
        let dir = TempStrDir::default();
        let uri = format!("file-object-store://{}", dir.as_str());
        let batch = record_batch!(
            (
                "x",
                Int32,
                [
                    Some(0),
                    Some(1),
                    None,
                    Some(3),
                    Some(4),
                    Some(5),
                    Some(6),
                    None
                ]
            ),
            ("y", Int32, [0, 1, 2, 3, 4, 5, 6, 7])
        )
        .unwrap();
        let mut dataset = Dataset::write(
            reader(batch),
            &uri,
            Some(WriteParams {
                max_rows_per_file: 4,
                max_rows_per_group: 4,
                enable_stable_row_ids: stable,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let params = ScalarIndexParams::for_builtin(kind.clone());
        let index_type = match kind {
            BuiltinIndexType::BTree => IndexType::BTree,
            BuiltinIndexType::Bitmap => IndexType::Bitmap,
            BuiltinIndexType::ZoneMap => IndexType::ZoneMap,
            BuiltinIndexType::BloomFilter => IndexType::BloomFilter,
            _ => unreachable!(),
        };
        let mut segments = Vec::new();
        for fragment_id in [0, 1] {
            segments.push(
                CreateIndexBuilder::new(&mut dataset, &["x"], index_type, &params)
                    .name("x_idx".to_string())
                    .fragments(vec![fragment_id])
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }
        dataset
            .commit_existing_index_segments("x_idx", "x", segments)
            .await
            .unwrap();
        let old_snapshot = dataset.clone();
        dataset
            .append(
                reader(
                    record_batch!(
                        ("x", Int32, [Some(8), Some(9), Some(10), Some(11)]),
                        ("y", Int32, [8, 9, 10, 11])
                    )
                    .unwrap(),
                ),
                None,
            )
            .await
            .unwrap();
        assert_eq!(dataset.fragments().len(), 3);
        // Keep the unsupported branch genuinely unindexed.
        let filters = [
            "x = 1",
            "x = 99",
            "x IN (1, 6, NULL)",
            "x IS NULL",
            "x >= 3 AND x < 6",
            "x = 1 OR y = 9",
            "x = 1 AND y % 2 = 1",
            "NOT (x = 1)",
            "x NOT IN (1, NULL)",
        ];
        let mut expected = Vec::new();
        for filter in filters {
            let batch = dataset
                .scan()
                .filter(filter)
                .unwrap()
                .use_scalar_index(false)
                .with_row_address()
                .project(&[] as &[&str])
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            let addresses = batch[ROW_ADDR]
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            expected.push(
                addresses
                    .values()
                    .iter()
                    .map(|address| RowAddress::from(*address).fragment_id())
                    .collect::<RoaringBitmap>(),
            );
        }
        let cold = reopen_without_data(&uri).await;
        for _ in 0..2 {
            for (filter, expected) in filters.into_iter().zip(&expected) {
                let candidates = query(&cold, filter).await;
                assert!(
                    expected.is_subset(&candidates.to_scan()),
                    "{kind:?} {stable} {filter}: {candidates:?} misses {expected:?}"
                );
                assert!(candidates.hits.is_disjoint(&candidates.unknown));
            }
        }
        let equality = query(&cold, "x = 1").await;
        assert!(equality.unknown.contains(2));
        if matches!(kind, BuiltinIndexType::BTree | BuiltinIndexType::Bitmap) {
            assert_eq!(equality.hits, RoaringBitmap::from_iter([0]));
            assert_eq!(equality.unknown, RoaringBitmap::from_iter([2]));
        }
        assert!(!query(&old_snapshot, "x = 1").await.to_scan().contains(2));
        // Check that the I/O barrier really rejects an ordinary data read.
        let error = cold.scan().try_into_batch().await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("fragment pruning attempted a data read")
        );
    }
    #[tokio::test]
    async fn test_boolean_partial_coverage() {
        let dir = TempStrDir::default();
        let uri = format!("file-object-store://{}", dir.as_str());
        let mut dataset = Dataset::write(
            reader(record_batch!(("x", Int32, [0, 1, 2, 3]), ("y", Int32, [0, 1, 2, 3])).unwrap()),
            &uri,
            Some(WriteParams {
                max_rows_per_file: 2,
                max_rows_per_group: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        dataset
            .create_index(&["x"], IndexType::Scalar, None, &params, false)
            .await
            .unwrap();
        dataset
            .append(
                reader(record_batch!(("x", Int32, [4, 5]), ("y", Int32, [4, 5])).unwrap()),
                None,
            )
            .await
            .unwrap();
        dataset
            .create_index(&["y"], IndexType::Scalar, None, &params, false)
            .await
            .unwrap();
        let dataset = reopen_without_data(&uri).await;
        let result = query(&dataset, "x = 0 AND y = 2").await;
        assert!(result.to_scan().is_empty());
        let result = query(&dataset, "x = 0 AND y = 4").await;
        assert_eq!(result.unknown, RoaringBitmap::from_iter([2]));
        assert!(result.hits.is_empty());
        let result = query(&dataset, "x = 0 OR y = 2").await;
        assert_eq!(result.hits, RoaringBitmap::from_iter([0, 1]));
        assert_eq!(result.unknown, RoaringBitmap::from_iter([2]));
        // Different rows in a fragment satisfy the two predicates; retaining
        // that fragment is intentional for fragment-level boolean algebra.
        assert_eq!(
            query(&dataset, "x = 0 AND y = 1").await.to_scan(),
            RoaringBitmap::from_iter([0])
        );
        assert!(query(&dataset, "false").await.to_scan().is_empty());
        assert_eq!(
            query(&dataset, "true").await.to_scan(),
            RoaringBitmap::from_iter([0, 1, 2])
        );
    }

    #[rstest]
    #[case::unrelated_field(false, false)]
    #[case::already_indexed(true, false)]
    #[case::stale(true, true)]
    #[tokio::test]
    async fn test_overlay_coverage(#[case] indexed_field: bool, #[case] newer: bool) {
        let mut dataset = Dataset::write(
            reader(record_batch!(("x", Int32, [1, 2]), ("y", Int32, [1, 2])).unwrap()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 1,
                max_rows_per_group: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["x"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();
        let field = dataset
            .schema()
            .field(if indexed_field { "x" } else { "y" })
            .unwrap()
            .id;
        let version = dataset.manifest.version;
        // No overlay file exists: pruning is required to inspect only its
        // metadata, including field identity and the index build version.
        let mut file = dataset.fragments()[1].files[0].clone();
        file.path = "unreadable-overlay.lance".into();
        file.fields = vec![field].into();
        let manifest = Arc::make_mut(&mut dataset.manifest);
        Arc::make_mut(&mut manifest.fragments)[1].overlays.push(
            lance_table::format::overlay::DataOverlayFile {
                data_file: file,
                coverage: lance_table::format::overlay::OverlayCoverage::dense(
                    RoaringBitmap::from_iter([0]),
                ),
                committed_version: if newer { version } else { 0 },
            },
        );
        let result = query(&dataset, "x = 99").await;
        assert!(result.hits.is_empty());
        assert_eq!(
            result.unknown,
            if newer && indexed_field {
                RoaringBitmap::from_iter([1])
            } else {
                RoaringBitmap::new()
            }
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_deleted_rows_without_deletion_reads(#[values(false, true)] stable: bool) {
        let dir = TempStrDir::default();
        let uri = format!("file-object-store://{}", dir.as_str());
        let mut dataset = Dataset::write(
            reader(record_batch!(("x", Int32, [0, 1, 2, 3])).unwrap()),
            &uri,
            Some(WriteParams {
                enable_stable_row_ids: stable,
                max_rows_per_file: 2,
                max_rows_per_group: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["x"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();
        dataset.delete("x = 0").await.unwrap();
        let dataset = reopen_without_data(&uri).await;
        assert_eq!(
            query(&dataset, "x = 3").await.to_scan(),
            RoaringBitmap::from_iter([1])
        );
        // An index hit for a tombstoned row is an allowed false positive.
        assert!(
            query(&dataset, "x = 0")
                .await
                .to_scan()
                .is_subset(&RoaringBitmap::from_iter([0]))
        );
    }

    #[tokio::test]
    async fn test_index_io_errors_propagate() {
        let dir = TempStrDir::default();
        let uri = format!("file-object-store://{}", dir.as_str());
        let mut dataset = Dataset::write(
            reader(record_batch!(("x", Int32, [1, 2])).unwrap()),
            &uri,
            None,
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["x"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();
        let proxy = Arc::new(FailingProxyStore::new());
        proxy.fail_when("get_opts", "/_indices/", "index read failure");
        let dataset = DatasetBuilder::from_uri(&uri)
            .with_read_params(ReadParams {
                store_options: Some(ObjectStoreParams {
                    object_store_wrapper: Some(proxy),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .load()
            .await
            .unwrap();
        let mut scanner = dataset.scan();
        scanner.filter("x = 1").unwrap();
        let error = scanner.candidate_fragments().await.unwrap_err();
        assert!(matches!(&error, Error::IO { .. }));
        assert!(error.to_string().contains("index read failure"));
    }
    #[tokio::test]
    async fn test_unknown_segment_coverage() {
        let mut dataset = Dataset::write(
            reader(record_batch!(("x", Int32, [0, 1])).unwrap()),
            "memory://",
            None,
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["x"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();
        let mut indices = dataset.load_indices().await.unwrap().as_ref().clone();
        let mut unknown = indices[0].clone();
        unknown.uuid = uuid::Uuid::new_v4();
        unknown.fragment_bitmap = None;
        indices.push(unknown);
        // Inject an additional segment whose coverage is absent. No index
        // files exist for it, so it must be retained without being searched.
        dataset
            .index_cache
            .insert_with_key(
                &crate::session::index_caches::IndexMetadataKey {
                    version: dataset.manifest.version,
                    store_identity: &dataset.object_store.store_prefix,
                    e_tag: dataset.manifest_location.e_tag.as_deref(),
                },
                Arc::new(indices),
            )
            .await;
        let result = query(&dataset, "x = 99").await;
        assert_eq!(result.unknown, *dataset.fragment_bitmap);
        assert!(result.hits.is_empty());
    }
    #[rstest]
    #[tokio::test]
    async fn test_null_fragments(
        #[values(BuiltinIndexType::BTree, BuiltinIndexType::ZoneMap)] kind: BuiltinIndexType,
    ) {
        let dataset_batch = record_batch!(("x", Int32, [None, None, Some(1), Some(2)])).unwrap();
        let mut dataset = Dataset::write(
            reader(dataset_batch),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 2,
                max_rows_per_group: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["x"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::for_builtin(kind),
                false,
            )
            .await
            .unwrap();
        assert_eq!(
            query(&dataset, "x IS NULL").await.to_scan(),
            RoaringBitmap::from_iter([0])
        );
        assert!(query(&dataset, "x = NULL").await.to_scan().is_empty());
    }
    #[rstest]
    #[tokio::test]
    async fn test_compacted_addresses(
        #[values(BuiltinIndexType::BTree, BuiltinIndexType::ZoneMap)] kind: BuiltinIndexType,
        #[values(false, true)] deferred: bool,
    ) {
        let dir = TempStrDir::default();
        let uri = format!("file-object-store://{}", dir.as_str());
        let mut dataset = Dataset::write(
            reader(record_batch!(("x", Int32, [0, 1, 2, 3])).unwrap()),
            &uri,
            Some(WriteParams {
                max_rows_per_file: 2,
                max_rows_per_group: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["x"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::for_builtin(kind),
                false,
            )
            .await
            .unwrap();
        crate::dataset::optimize::compact_files(
            &mut dataset,
            crate::dataset::optimize::CompactionOptions {
                target_rows_per_fragment: 4,
                defer_index_remap: deferred,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        let dataset = reopen_without_data(&uri).await;
        assert_eq!(dataset.fragments().len(), 1);
        assert!(!dataset.fragment_bitmap.contains(0));
        assert_eq!(
            query(&dataset, "x = 3").await.to_scan(),
            *dataset.fragment_bitmap
        );
        let missing = query(&dataset, "x = 99").await;
        assert!(missing.hits.is_empty());
        assert!(missing.unknown.is_subset(&dataset.fragment_bitmap));
    }

    #[tokio::test]
    async fn test_scope_empty_and_unsupported() {
        let mut dataset = Dataset::write(
            reader(record_batch!(("x", Int32, [0, 1, 2, 3])).unwrap()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 2,
                max_rows_per_group: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["x"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
                false,
            )
            .await
            .unwrap();
        assert_eq!(
            query(&dataset, "x = 99").await.unknown,
            *dataset.fragment_bitmap
        );
        let mut scanner = dataset.scan();
        scanner.with_fragments(vec![dataset.fragments()[1].clone()]);
        scanner.filter("x = 99").unwrap();
        assert_eq!(
            scanner.candidate_fragments().await.unwrap().unknown,
            RoaringBitmap::from_iter([1])
        );
        dataset.delete("x < 2").await.unwrap();
        let candidates = dataset.scan().candidate_fragments().await.unwrap();
        assert!(!candidates.to_scan().contains(0));
        assert!(candidates.to_scan().contains(1));
    }
}
