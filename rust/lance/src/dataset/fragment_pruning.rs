// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Conservative fragment pruning using scalar indices, without dataset reads.

use std::sync::Arc;

use arrow_schema::{DataType, Schema};
use async_recursion::async_recursion;
use datafusion::common::{DFSchema, ScalarValue};
use datafusion_expr::{Expr, ExprSchemable, Operator};
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use lance_datafusion::planner::Planner;
use lance_index::metrics::MetricsCollector;
use lance_index::scalar::FragmentMapper;
use lance_index::scalar::expression::{ScalarIndexExpr, ScalarIndexSearch, apply_scalar_indices};
use lance_select::{RowAddrSelection, RowAddrTreeMap};
use lance_table::format::RowIdMeta;
use lance_table::rowids::segment::U64Segment;
use lance_table::rowids::{RowIdSequence, read_row_ids};
use roaring::RoaringBitmap;
use tokio::sync::OnceCell;

use super::Dataset;
use super::overlay::{collect_overlay_stale_frags, overlaid_fragments};
use crate::index::scalar_logical::load_named_scalar_segments;
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt, ScalarIndexInfo};
use crate::session::caches::RowIdSequenceKey;

/// Disjoint candidate sets for one dataset snapshot.
///
/// Neither set guarantees a matching live row. Their union contains every
/// fragment with a matching row, including fragments no usable index covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentCandidates {
    /// Fragments retained by index predicates, without an unresolved condition.
    pub indexed_candidates: RoaringBitmap,
    /// Candidates retained because coverage, mappings, or predicate support is missing.
    pub unresolved_candidates: RoaringBitmap,
}

impl FragmentCandidates {
    /// All fragments that must be considered by a subsequent query.
    pub fn all(&self) -> RoaringBitmap {
        &self.indexed_candidates | &self.unresolved_candidates
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

/// Runs scalar fragment pruning on a fixed snapshot without opening data files.
///
/// Index files may be read in full for broad predicates. Unsupported predicates,
/// stale index coverage, and unavailable resident row-ID mappings retain
/// fragments conservatively. Deletion vectors are not read.
///
/// ```
/// # use lance::{Dataset, Result};
/// # use lance::dataset::fragment_pruning::FragmentPruner;
/// # use lance_index::metrics::NoOpMetricsCollector;
/// # use datafusion_expr::{col, lit};
/// # async fn example(dataset: &Dataset) -> Result<()> {
/// let candidates = FragmentPruner::new(dataset)
///     .prune(col("id").eq(lit(42)), &NoOpMetricsCollector)
///     .await?;
/// let fragment_ids = candidates.all();
/// # Ok(())
/// # }
/// ```
pub struct FragmentPruner<'a> {
    dataset: &'a Dataset,
    row_ids: OnceCell<ResidentRowIds>,
}

impl<'a> FragmentPruner<'a> {
    /// Bind pruning and any resident identifier mapping to this snapshot.
    pub fn new(dataset: &'a Dataset) -> Self {
        Self {
            dataset,
            row_ids: OnceCell::new(),
        }
    }

    /// Return a conservative superset for a scalar filter.
    ///
    /// Performs expression validation and scalar-index I/O only. Storage errors
    /// propagate; they are never interpreted as an empty candidate set.
    pub async fn prune(
        &self,
        filter: Expr,
        metrics: &dyn MetricsCollector,
    ) -> Result<FragmentCandidates> {
        let schema = Arc::new(Schema::from(self.dataset.schema()));
        let planner = Planner::new(schema.clone());
        let filter = planner.optimize_expr(filter)?;
        let data_type = filter.get_type(&DFSchema::try_from(schema.as_ref().clone())?)?;
        if !matches!(data_type, DataType::Boolean | DataType::Null) {
            return Err(Error::invalid_input(format!(
                "Fragment pruning requires a Boolean filter, got {data_type} for {filter}"
            )));
        }
        let info = self.dataset.scalar_index_info().await?;
        let result = self.evaluate(&filter, &info, metrics).await?;
        Ok(FragmentCandidates {
            indexed_candidates: &result.possible - &result.unresolved,
            unresolved_candidates: result.unresolved,
        })
    }

    #[async_recursion]
    async fn evaluate(
        &self,
        expr: &Expr,
        info: &ScalarIndexInfo,
        metrics: &dyn MetricsCollector,
    ) -> Result<Candidates> {
        let scope = self.dataset.fragment_bitmap.as_ref();
        match expr {
            Expr::BinaryExpr(binary) if binary.op == Operator::And => {
                let left = self.evaluate(&binary.left, info, metrics).await?;
                if left.possible.is_empty() {
                    return Ok(left);
                }
                Ok(left.and(self.evaluate(&binary.right, info, metrics).await?))
            }
            Expr::BinaryExpr(binary) if binary.op == Operator::Or => Ok(self
                .evaluate(&binary.left, info, metrics)
                .await?
                .or(self.evaluate(&binary.right, info, metrics).await?)),
            Expr::Literal(ScalarValue::Boolean(value), _) => Ok(Candidates {
                possible: if *value == Some(true) {
                    scope.clone()
                } else {
                    RoaringBitmap::new()
                },
                unresolved: RoaringBitmap::new(),
            }),
            Expr::Literal(ScalarValue::Null, _) => Ok(Candidates {
                possible: RoaringBitmap::new(),
                unresolved: RoaringBitmap::new(),
            }),
            _ => {
                let parsed = apply_scalar_indices(expr.clone(), info)?;
                let Some(query) = parsed.scalar_query else {
                    return Ok(Candidates::unknown(scope));
                };
                let mut result = self.evaluate_index(&query, metrics).await?;
                if parsed.refine_expr.is_some() {
                    result.unresolved = result.possible.clone();
                }
                Ok(result)
            }
        }
    }

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
            ScalarIndexExpr::Not(_) => Ok(Candidates::unknown(&self.dataset.fragment_bitmap)),
            ScalarIndexExpr::Query(search) => self.search(search, metrics).await,
        }
    }

    async fn search(
        &self,
        search: &ScalarIndexSearch,
        metrics: &dyn MetricsCollector,
    ) -> Result<Candidates> {
        let scope = self.dataset.fragment_bitmap.as_ref();
        // The logical loader filters out unknown coverage. Inspect it first:
        // absence of a bitmap must not become evidence of an empty segment.
        if self
            .dataset
            .load_indices_by_name(&search.index_name)
            .await?
            .iter()
            .any(|segment| segment.fragment_bitmap.is_none())
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
            unresolved |= &stale;
            coverage -= stale;
            if coverage.is_empty() {
                continue;
            }
            let index = self
                .dataset
                .open_scalar_index(&search.column, &segment.uuid, metrics)
                .await?;
            let resident = if self.dataset.manifest.uses_stable_row_ids()
                && !index.results_are_row_addresses()
            {
                let resident = self
                    .row_ids
                    .get_or_try_init(|| self.load_resident_row_ids())
                    .await?;
                unresolved |= &coverage & &resident.unresolved;
                coverage -= &resident.unresolved;
                Some(resident)
            } else {
                None
            };
            if coverage.is_empty() {
                continue;
            }
            let mapper = SnapshotFragmentMapper {
                resident,
                scope: &coverage,
            };
            match index
                .search_fragments(search.query.as_ref(), &mapper, metrics)
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

    async fn load_resident_row_ids(&self) -> Result<ResidentRowIds> {
        let mut sequences = Vec::new();
        let mut unresolved = RoaringBitmap::new();
        for fragment in self.dataset.fragments().iter() {
            let fragment_id = u32::try_from(fragment.id).map_err(|_| {
                Error::invalid_input(format!("Fragment ID {} exceeds u32", fragment.id))
            })?;
            let sequence = match &fragment.row_id_meta {
                Some(RowIdMeta::Inline(bytes)) => Some(Arc::new(read_row_ids(bytes)?)),
                Some(meta @ RowIdMeta::Column) => {
                    self.dataset
                        .metadata_cache
                        .get_with_key(&RowIdSequenceKey {
                            fragment_id: fragment.id,
                            row_id_meta: meta,
                            lineage_file: fragment
                                .row_lineage_file(lance_table::format::ROW_ID_FIELD_ID)?,
                        })
                        .await
                }
                None => None,
            };
            if let Some(sequence) = sequence {
                sequences.push((fragment_id, sequence));
            } else {
                unresolved.insert(fragment_id);
            }
        }
        Ok(ResidentRowIds {
            sequences,
            unresolved,
        })
    }
}

struct ResidentRowIds {
    sequences: Vec<(u32, Arc<RowIdSequence>)>,
    unresolved: RoaringBitmap,
}

struct SnapshotFragmentMapper<'a> {
    // None means entries are physical addresses, including non-stable row IDs.
    resident: Option<&'a ResidentRowIds>,
    scope: &'a RoaringBitmap,
}

impl FragmentMapper for SnapshotFragmentMapper<'_> {
    fn map_ids(&self, ids: &[u64]) -> Result<RoaringBitmap> {
        let Some(resident) = self.resident else {
            return Ok(ids
                .iter()
                .map(|id| RowAddress::from(*id).fragment_id())
                .filter(|id| self.scope.contains(*id))
                .collect());
        };
        // Keep every possible owner, including tombstoned duplicates. Loading a
        // deletion vector to select a single owner would violate index-only I/O.
        Ok(resident
            .sequences
            .iter()
            .filter_map(|(fragment, sequence)| {
                (self.scope.contains(*fragment)
                    && ids.iter().any(|id| {
                        sequence
                            .segments()
                            .iter()
                            .any(|segment| segment.contains(*id))
                    }))
                .then_some(*fragment)
            })
            .collect())
    }

    fn map_rows(&self, rows: &RowAddrTreeMap) -> Result<RoaringBitmap> {
        let Some(resident) = self.resident else {
            return Ok(rows
                .iter()
                .filter_map(|(fragment, selection)| {
                    let nonempty = match selection {
                        RowAddrSelection::Full => true,
                        RowAddrSelection::Partial(bitmap) => !bitmap.is_empty(),
                    };
                    (nonempty && self.scope.contains(*fragment)).then_some(*fragment)
                })
                .collect());
        };
        let mut fragments = RoaringBitmap::new();
        for (fragment, sequence) in &resident.sequences {
            if !self.scope.contains(*fragment) {
                continue;
            }
            let intersects = sequence.segments().iter().any(|segment| {
                let Some(bounds) = segment.range() else {
                    return false;
                };
                rows.iter().any(|(prefix, selection)| {
                    // This is a container prefix of a logical row ID, not a
                    // physical fragment ID. Compare compressed ranges first.
                    let base = u64::from(*prefix) << 32;
                    let start = (*bounds.start()).max(base);
                    let end = (*bounds.end()).min(base | u64::from(u32::MAX));
                    if start > end {
                        return false;
                    }
                    match selection {
                        RowAddrSelection::Full => true,
                        RowAddrSelection::Partial(bitmap) => {
                            let offsets = (start as u32)..=(end as u32);
                            if matches!(segment, U64Segment::Range(_)) {
                                bitmap.range_cardinality(offsets) > 0
                            } else {
                                bitmap
                                    .range(offsets)
                                    .any(|offset| segment.contains(base | u64::from(offset)))
                            }
                        }
                    }
                })
            });
            if intersects {
                fragments.insert(*fragment);
            }
        }
        Ok(fragments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{RecordBatch, RecordBatchIterator, UInt64Array, record_batch};
    use datafusion_expr::{col, lit};
    use lance_core::ROW_ADDR;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_index::IndexType;
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
    use lance_io::object_store::ObjectStoreParams;
    use rstest::rstest;

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
        let planner = Planner::new(Arc::new(Schema::from(dataset.schema())));
        FragmentPruner::new(dataset)
            .prune(planner.parse_filter(filter).unwrap(), &NoOpMetricsCollector)
            .await
            .unwrap()
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
        #[values(
            BuiltinIndexType::BTree,
            BuiltinIndexType::Bitmap,
            BuiltinIndexType::ZoneMap,
            BuiltinIndexType::BloomFilter
        )]
        kind: BuiltinIndexType,
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
                    expected.is_subset(&candidates.all()),
                    "{kind:?} {stable} {filter}: {candidates:?} misses {expected:?}"
                );
                assert!(
                    candidates
                        .indexed_candidates
                        .is_disjoint(&candidates.unresolved_candidates)
                );
            }
        }
        let equality = query(&cold, "x = 1").await;
        assert!(equality.unresolved_candidates.contains(2));
        if matches!(kind, BuiltinIndexType::BTree | BuiltinIndexType::Bitmap) {
            assert_eq!(equality.indexed_candidates, RoaringBitmap::from_iter([0]));
            assert_eq!(
                equality.unresolved_candidates,
                RoaringBitmap::from_iter([2])
            );
        }
        assert!(!query(&old_snapshot, "x = 1").await.all().contains(2));
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
        assert!(result.all().is_empty());
        let result = query(&dataset, "x = 0 AND y = 4").await;
        assert_eq!(result.unresolved_candidates, RoaringBitmap::from_iter([2]));
        assert!(result.indexed_candidates.is_empty());
        let result = query(&dataset, "x = 0 OR y = 2").await;
        assert_eq!(result.indexed_candidates, RoaringBitmap::from_iter([0, 1]));
        assert_eq!(result.unresolved_candidates, RoaringBitmap::from_iter([2]));
        // Different rows in a fragment satisfy the two predicates; retaining
        // that fragment is intentional for fragment-level boolean algebra.
        assert_eq!(
            query(&dataset, "x = 0 AND y = 1").await.all(),
            RoaringBitmap::from_iter([0])
        );
        assert!(query(&dataset, "false").await.all().is_empty());
        assert_eq!(
            query(&dataset, "true").await.all(),
            RoaringBitmap::from_iter([0, 1, 2])
        );
    }

    #[tokio::test]
    async fn test_unindexed_and_invalid_predicates() {
        let dataset = Dataset::write(
            reader(record_batch!(("x", Int32, [1, 2])).unwrap()),
            "memory://",
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            query(&dataset, "x = 99").await.unresolved_candidates,
            RoaringBitmap::from_iter([0])
        );
        let error = FragmentPruner::new(&dataset)
            .prune(lit(1), &NoOpMetricsCollector)
            .await
            .unwrap_err();
        assert!(matches!(&error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("requires a Boolean filter"));
        assert!(
            FragmentPruner::new(&dataset)
                .prune(col("missing").eq(lit(1)), &NoOpMetricsCollector)
                .await
                .is_err()
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
        assert!(result.indexed_candidates.is_empty());
        assert_eq!(
            result.unresolved_candidates,
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
                &ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
                false,
            )
            .await
            .unwrap();
        dataset.delete("x = 0").await.unwrap();
        let dataset = reopen_without_data(&uri).await;
        assert_eq!(
            query(&dataset, "x = 3").await.all(),
            RoaringBitmap::from_iter([1])
        );
        // An index hit for a tombstoned row is an allowed false positive.
        assert!(
            query(&dataset, "x = 0")
                .await
                .all()
                .is_subset(&RoaringBitmap::from_iter([0]))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_compacted_stable_ids(#[values(false, true)] spill: bool) {
        let dir = TempStrDir::default();
        let uri = format!("file-object-store://{}", dir.as_str());
        let mut dataset = Dataset::write(
            reader(record_batch!(("x", Int32, [0, 1, 2, 3])).unwrap()),
            &uri,
            Some(WriteParams {
                enable_stable_row_ids: true,
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
        if spill {
            dataset
                .update_config([
                    (super::super::rowids::SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
                    (
                        super::super::rowids::INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY,
                        "0",
                    ),
                ])
                .await
                .unwrap();
        }
        crate::dataset::optimize::compact_files(
            &mut dataset,
            crate::dataset::optimize::CompactionOptions {
                target_rows_per_fragment: 4,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(dataset.fragments().len(), 1);
        assert_eq!(
            matches!(dataset.fragments()[0].row_id_meta, Some(RowIdMeta::Column)),
            spill
        );
        let dataset = reopen_without_data(&uri).await;
        let result = query(&dataset, "x = 3").await;
        assert_eq!(result.all(), *dataset.fragment_bitmap);
        if spill {
            assert_eq!(result.unresolved_candidates, *dataset.fragment_bitmap);
            // Once a legitimate caller has warmed the lineage cache, a new
            // pruning operation can use it without reopening the data file.
            let warmer = Dataset::open(&uri).await.unwrap();
            let fragment = &dataset.fragments()[0];
            let sequence =
                super::super::rowids::load_row_id_sequence(&warmer, &warmer.fragments()[0])
                    .await
                    .unwrap();
            dataset
                .metadata_cache
                .insert_with_key(
                    &RowIdSequenceKey {
                        fragment_id: fragment.id,
                        row_id_meta: fragment.row_id_meta.as_ref().unwrap(),
                        lineage_file: fragment
                            .row_lineage_file(lance_table::format::ROW_ID_FIELD_ID)
                            .unwrap(),
                    },
                    sequence,
                )
                .await;
            assert_eq!(
                query(&dataset, "x = 3").await.indexed_candidates,
                *dataset.fragment_bitmap
            );
        } else {
            assert_eq!(result.indexed_candidates, *dataset.fragment_bitmap);
            assert!(query(&dataset, "x = 99").await.all().is_empty());
        }
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
        let error = FragmentPruner::new(&dataset)
            .prune(col("x").eq(lit(1)), &NoOpMetricsCollector)
            .await
            .unwrap_err();
        assert!(matches!(&error, Error::IO { .. }));
        assert!(error.to_string().contains("index read failure"));
    }
    #[test]
    fn test_fragment_mapper_domains_and_sparse_ids() {
        let scope = RoaringBitmap::from_iter([4, 9]);
        let resident = ResidentRowIds {
            sequences: vec![
                (
                    4,
                    Arc::new(RowIdSequence::try_from_iter([0, 2, 100]).unwrap()),
                ),
                (
                    9,
                    Arc::new(RowIdSequence::try_from_iter([1u64 << 40, (1u64 << 40) + 2]).unwrap()),
                ),
            ],
            unresolved: RoaringBitmap::new(),
        };
        let logical = SnapshotFragmentMapper {
            resident: Some(&resident),
            scope: &scope,
        };
        assert_eq!(
            logical.map_ids(&[2]).unwrap(),
            RoaringBitmap::from_iter([4])
        );
        assert!(
            logical
                .map_rows(&RowAddrTreeMap::from_iter([1]))
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            logical
                .map_rows(&RowAddrTreeMap::from_iter([(1u64 << 40) + 2]))
                .unwrap(),
            RoaringBitmap::from_iter([9])
        );
        let physical = SnapshotFragmentMapper {
            resident: None,
            scope: &scope,
        };
        let mut rows = RowAddrTreeMap::new();
        rows.insert_fragment(9);
        rows.insert_bitmap(4, RoaringBitmap::new());
        assert_eq!(
            physical.map_rows(&rows).unwrap(),
            RoaringBitmap::from_iter([9])
        );
        assert!(physical.map_ids(&[2]).unwrap().is_empty());
    }

    #[rstest]
    #[tokio::test]
    async fn test_null_fragments(
        #[values(
            BuiltinIndexType::BTree,
            BuiltinIndexType::Bitmap,
            BuiltinIndexType::ZoneMap,
            BuiltinIndexType::BloomFilter
        )]
        kind: BuiltinIndexType,
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
            query(&dataset, "x IS NULL").await.all(),
            RoaringBitmap::from_iter([0])
        );
        assert!(query(&dataset, "x = NULL").await.all().is_empty());
    }
    #[rstest]
    #[tokio::test]
    async fn test_compacted_addresses(
        #[values(BuiltinIndexType::BTree, BuiltinIndexType::Bitmap)] kind: BuiltinIndexType,
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
            query(&dataset, "x = 3").await.all(),
            *dataset.fragment_bitmap
        );
        assert!(query(&dataset, "x = 99").await.all().is_empty());
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
        assert_eq!(result.unresolved_candidates, *dataset.fragment_bitmap);
        assert!(result.indexed_candidates.is_empty());
    }
}
