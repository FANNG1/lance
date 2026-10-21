// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;
use std::vec;

use crate::index::vector::VectorIndexParams;
use lance_arrow::json::{ARROW_JSON_EXT_NAME, JsonArray, is_arrow_json_field, json_field};
use lance_arrow::{ARROW_EXT_NAME_KEY, FixedSizeListArrayExt};

use crate::index::DatasetIndexExt;
use arrow::compute::concat_batches;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, LargeListArray, ListArray, StructArray};
use arrow_array::{Float32Array, Int32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_array::{Int64Array, UInt64Array};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema, SchemaRef};
use futures::TryStreamExt;
use lance_arrow::SchemaExt;
use lance_core::ROW_ADDR;
use lance_core::cache::LanceCache;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tempfile::TempStrDir;
use lance_encoding::decoder::DecoderPlugins;
use lance_file::reader::{FileReader, FileReaderOptions, describe_encoding};
use lance_file::version::LanceFileVersion;
use lance_index::scalar::inverted::{
    SCORE_FIELD,
    query::{FtsQuery, MatchQuery, Operator, PhraseQuery},
    tokenizer::InvertedIndexParams,
};
use lance_index::scalar::minhash_lsh::MinHashQuery;
use lance_index::scalar::{BuiltinIndexType, FullTextSearchQuery, ScalarIndexParams};
use lance_index::{IndexType, vector::DIST_COL};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_linalg::distance::MetricType;
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::dataset::DATA_DIR;
use crate::dataset::NewColumnTransform;
use crate::dataset::scanner::{DatasetRecordBatchStream, FragmentCandidates, QueryFilter};
use crate::dataset::write::{WriteMode, WriteParams};
use crate::{Dataset, Error};
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::{DEFAULT_QUERY_PARALLELISM, Query};
use pretty_assertions::assert_eq;
use rstest::rstest;

/// A null struct must not read back as a valid struct with null children.
///
/// A scan merges the per-column batches with `lance_arrow::merge`, which used to read an all-null
/// validity buffer as "this side carries no validity" and drop it. A filter that selects only null
/// rows leaves exactly that shape, so the scan reported those rows as valid while `IS NULL`
/// counted them as null. The dataset is created empty and then appended to because that is the
/// path this was found on, and the version is pinned because 2.0 does not encode struct validity.
#[tokio::test]
async fn test_filtered_scan_preserves_nullable_struct_validity() {
    let struct_fields = Fields::from(vec![
        ArrowField::new("a", DataType::Int64, true),
        ArrowField::new("b", DataType::Utf8, true),
    ]);
    let item_field = Arc::new(ArrowField::new(
        "item",
        DataType::Struct(struct_fields.clone()),
        true,
    ));
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::UInt64, false),
        ArrowField::new("s", DataType::Struct(struct_fields.clone()), true),
        ArrowField::new("l", DataType::List(item_field.clone()), true),
    ]));

    let empty = RecordBatch::new_empty(schema.clone());
    let reader = RecordBatchIterator::new([Ok(empty)], schema.clone());
    let mut dataset = Dataset::write(
        reader,
        "memory://",
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..WriteParams::default()
        }),
    )
    .await
    .unwrap();

    // Rows 100 and 177 are null in both nested columns while their children still carry values,
    // so losing the top-level validity turns them into valid values instead of null ones.
    let validity = NullBuffer::from(vec![false, true, false]);
    let structs = StructArray::new(
        struct_fields.clone(),
        vec![
            Arc::new(Int64Array::from(vec![Some(10), Some(11), Some(12)])),
            Arc::new(StringArray::from(vec![Some("x"), Some("y"), Some("z")])),
        ],
        Some(validity.clone()),
    );
    let items = StructArray::new(
        struct_fields.clone(),
        vec![
            Arc::new(Int64Array::from(vec![Some(20), Some(21), Some(22)])),
            Arc::new(StringArray::from(vec![Some("p"), Some("q"), Some("r")])),
        ],
        None,
    );
    let lists = ListArray::new(
        item_field,
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 1, 2, 3])),
        Arc::new(items),
        Some(validity),
    );
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![100, 116, 177])),
            Arc::new(structs),
            Arc::new(lists),
        ],
    )
    .unwrap();
    dataset
        .append(
            Box::new(RecordBatchIterator::new([Ok(batch)], schema)),
            None,
        )
        .await
        .unwrap();

    for (id, expected_is_null) in [(100, true), (116, false), (177, true)] {
        let mut scan = dataset.scan();
        scan.filter(&format!("id = {id}")).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        let structs = batch["s"].as_struct();
        assert_eq!(structs.is_null(0), expected_is_null, "s, row id {id}");
        // A null struct masks its children, so both levels have to agree.
        assert_eq!(
            structs.column(0).is_null(0),
            expected_is_null,
            "s.a, row id {id}"
        );
        assert_eq!(
            batch["l"].as_list::<i32>().is_null(0),
            expected_is_null,
            "l, row id {id}"
        );
    }

    assert_eq!(
        dataset
            .count_rows(Some("s IS NULL".to_owned()))
            .await
            .unwrap(),
        2
    );
    assert_eq!(
        dataset
            .count_rows(Some("l IS NULL".to_owned()))
            .await
            .unwrap(),
        2
    );

    // A struct column added as all-nulls reaches the same merge through schema evolution, where
    // every row is null and there is no other side to recover the validity from.
    dataset
        .add_columns(
            NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "t",
                DataType::Struct(struct_fields),
                true,
            )]))),
            None,
            None,
        )
        .await
        .unwrap();
    let mut scan = dataset.scan();
    scan.filter("id = 116").unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    assert!(batch["t"].as_struct().is_null(0));
    assert_eq!(
        dataset
            .count_rows(Some("t IS NULL".to_owned()))
            .await
            .unwrap(),
        3
    );
}

#[tokio::test]
async fn test_scan_wide_fixed_size_list_at_batch_boundary() {
    const DIM_A: usize = 140_000;
    const DIM_B: usize = 4_096;
    const SHORT_ROWS: usize = 68;
    const LONG_ROWS: usize = 128;

    fn make_batch(schema: SchemaRef, rows: usize, base: usize) -> RecordBatch {
        let values_a = Float32Array::from_iter_values(
            (0..rows * DIM_A).map(|idx| ((idx + base) % 1009) as f32 / 1009.0),
        );
        let values_b = Float32Array::from_iter_values(
            (0..rows * DIM_B).map(|idx| ((idx + base) % 251) as f32 / 251.0),
        );
        let arr_a = FixedSizeListArray::try_new_from_values(values_a, DIM_A as i32).unwrap();
        let arr_b = FixedSizeListArray::try_new_from_values(values_b, DIM_B as i32).unwrap();
        RecordBatch::try_new(schema, vec![Arc::new(arr_a), Arc::new(arr_b)]).unwrap()
    }

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(
            "a",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                DIM_A as i32,
            ),
            true,
        ),
        ArrowField::new(
            "b",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                DIM_B as i32,
            ),
            true,
        ),
    ]));

    let batches = vec![
        make_batch(schema.clone(), SHORT_ROWS, 0),
        make_batch(schema.clone(), LONG_ROWS, 17),
    ];
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let write_params = WriteParams {
        data_storage_version: Some(LanceFileVersion::V2_1),
        ..WriteParams::default()
    };
    let dir = tempfile::tempdir().unwrap();
    let dataset = Dataset::write(reader, dir.path().to_str().unwrap(), Some(write_params))
        .await
        .unwrap();

    // The first column splits into 9 read chunks. The second column is a
    // higher-priority request that can reserve the remaining buffer while the
    // first column is still awaited.
    let mut scanner = dataset.scan();
    scanner.io_buffer_size(70 * 1024 * 1024);
    scanner
        .limit(Some(LONG_ROWS as i64), Some(SHORT_ROWS as i64))
        .unwrap();
    let mut stream = tokio::time::timeout(
        std::time::Duration::from_secs(20),
        scanner.try_into_stream(),
    )
    .await
    .expect("stream creation timed out")
    .unwrap();
    let batch = tokio::time::timeout(std::time::Duration::from_secs(20), stream.try_next())
        .await
        .expect("first batch timed out")
        .unwrap()
        .unwrap();

    assert_eq!(batch.num_rows(), LONG_ROWS);
}

#[tokio::test]
async fn test_vector_filter_fts_search() {
    let dataset = prepare_query_filter_dataset().await;
    let schema: ArrowSchema = dataset.schema().into();

    let query_vector = Arc::new(Float32Array::from(vec![300f32, 300f32, 300f32, 300f32]));
    let vector_query = Query {
        column: "vector".to_string(),
        key: query_vector,
        k: 5,
        lower_bound: None,
        upper_bound: None,
        minimum_nprobes: 20,
        maximum_nprobes: None,
        search_effort: 0.5,
        ef: None,
        refine_factor: None,
        metric_type: Some(MetricType::L2),
        use_index: true,
        query_parallelism: DEFAULT_QUERY_PARALLELISM,
        dist_q_c: 0.0,
        approx_mode: Default::default(),
    };

    // Case 1: search with prefilter=true, query_filter=vector([300,300,300,300])
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new("text".to_string()))
        .unwrap()
        .prefilter(true)
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300, 299],
    )
    .await;

    // Case 2: search with prefilter=true, query_filter=vector([300,300,300,300]), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new("text".to_string()))
        .unwrap()
        .prefilter(true)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300],
    )
    .await;

    // Case 3: search with prefilter=true, phrase query, query_filter=vector([300,300,300,300])
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Phrase(
            PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
        )))
        .unwrap()
        .prefilter(true)
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[299, 300],
    )
    .await;

    // Case 4: search with prefilter=true, phrase query, query_filter=vector([300,300,300,300]), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Phrase(
            PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
        )))
        .unwrap()
        .prefilter(true)
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .filter("category='geography'")
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300],
    )
    .await;

    // Case 5: search with prefilter=false, phrase query, query_filter=vector([300,300,300,300])
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Phrase(
            PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
        )))
        .unwrap()
        .prefilter(false)
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300, 299, 255, 254, 253],
    )
    .await;

    // Case 6: search with prefilter=false, phrase query, query_filter=vector([300,300,300,300]), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Phrase(
            PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
        )))
        .unwrap()
        .prefilter(false)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300, 255],
    )
    .await;
}

#[tokio::test]
async fn test_fts_filter_vector_search() {
    let dataset = prepare_query_filter_dataset().await;
    let schema: ArrowSchema = dataset.schema().into();

    // Assert filtering and exact distance order without depending on PQ's
    // approximate distances, which can tie for adjacent vectors.
    // Case 1: search with prefilter=true, query_filter=match("text")
    let query_vector = Float32Array::from(vec![300f32, 300f32, 300f32, 300f32]);
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .refine(1)
        .prefilter(true)
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300, 299, 255, 254, 253],
    )
    .await;

    // Case 2: search with prefilter=true, query_filter=match("text"), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .refine(1)
        .prefilter(true)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300, 255, 252, 249, 246],
    )
    .await;

    // Case 3: search with prefilter=false, query_filter=match("text")
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .refine(1)
        .prefilter(false)
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300, 299],
    )
    .await;

    // Case 4: search with prefilter=false, query_filter=match("text"), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .refine(1)
        .prefilter(false)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300],
    )
    .await;

    // Case 5: search with prefilter=false, query_filter=phrase("text")
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(false)
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new_query(
            FtsQuery::Phrase(
                PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
            ),
        )))
        .unwrap()
        .try_into_stream()
        .await;
    assert!(stream.is_err());

    // Case 6: search with prefilter=false, query_filter=phrase("text")
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(false)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new_query(
            FtsQuery::Phrase(
                PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
            ),
        )))
        .unwrap()
        .try_into_stream()
        .await;
    assert!(stream.is_err());
}

#[rstest]
#[case::list(false)]
#[case::large_list(true)]
#[tokio::test]
async fn test_fts_list_postfilter_vector_search(#[case] is_large_list: bool) {
    async fn indexed_ids(dataset: &Dataset, query: FullTextSearchQuery) -> Vec<i32> {
        let result = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let mut ids = result["id"]
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec();
        ids.sort_unstable();
        ids
    }

    async fn postfilter_ids(dataset: &Dataset, query: FullTextSearchQuery) -> Vec<i32> {
        let query_vector = Float32Array::from(vec![0.0, 0.0]);
        let mut scanner = dataset.scan();
        scanner
            .nearest("vector", &query_vector, 5)
            .unwrap()
            .prefilter(false)
            .filter_query(QueryFilter::Fts(query))
            .unwrap();
        let plan = scanner.explain_plan(false).await.unwrap();
        let post_filter_position = plan
            .find("FlatMatchFilter: column=docs")
            .expect("expected FTS to run as a flat match filter");
        let vector_search_position = plan
            .find("ANNSubIndex")
            .expect("expected the query to use the vector index");
        assert!(
            post_filter_position < vector_search_position,
            "expected FTS to wrap the vector search as a post-filter, got:\n{plan}"
        );
        let result = scanner.try_into_batch().await.unwrap();
        let mut ids = result["id"]
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec();
        ids.sort_unstable();
        ids
    }

    fn match_query(terms: &str, operator: Operator) -> FullTextSearchQuery {
        FullTextSearchQuery::new_query(FtsQuery::Match(
            MatchQuery::new(terms.to_owned())
                .with_column(Some("docs".to_owned()))
                .with_operator(operator),
        ))
    }

    let item_field = Arc::new(ArrowField::new("item", DataType::Utf8, true));
    let values = Arc::new(StringArray::from(vec![
        Some("target"),
        Some("alpha"),
        Some("beta"),
        Some(""),
        None,
        Some("target"),
    ])) as ArrayRef;
    let validity = Some(NullBuffer::from(vec![true, true, true, true, false]));
    let docs: ArrayRef = if is_large_list {
        Arc::new(LargeListArray::new(
            item_field.clone(),
            OffsetBuffer::new(ScalarBuffer::from(vec![0_i64, 2, 3, 3, 6, 6])),
            values,
            validity,
        ))
    } else {
        Arc::new(ListArray::new(
            item_field,
            OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 3, 3, 6, 6])),
            values,
            validity,
        ))
    };
    let vectors = FixedSizeListArray::try_new_from_values(
        Float32Array::from(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]),
        2,
    )
    .unwrap();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("vector", vectors.data_type().clone(), false),
        ArrowField::new("docs", docs.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..5)),
            Arc::new(vectors),
            docs,
        ],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 3);

    dataset
        .create_index(
            &["docs"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            None,
            &VectorIndexParams::ivf_flat(1, MetricType::L2),
            true,
        )
        .await
        .unwrap();

    let query = match_query("target", Operator::Or);
    assert_eq!(indexed_ids(&dataset, query.clone()).await, [0, 3]);
    assert_eq!(postfilter_ids(&dataset, query).await, [0, 3]);

    let query = match_query("target alpha", Operator::And);
    assert_eq!(indexed_ids(&dataset, query.clone()).await, [0]);
    assert_eq!(postfilter_ids(&dataset, query).await, [0]);

    let query = match_query("target missing", Operator::And);
    assert!(indexed_ids(&dataset, query.clone()).await.is_empty());
    assert!(postfilter_ids(&dataset, query).await.is_empty());

    dataset
        .create_index(
            &["docs"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default().base_tokenizer("raw".to_owned()),
            true,
        )
        .await
        .unwrap();
    let query = match_query("target", Operator::Or);
    assert_eq!(indexed_ids(&dataset, query.clone()).await, [3]);
    assert_eq!(postfilter_ids(&dataset, query).await, [3]);

    dataset
        .create_index(
            &["docs"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::code()
                .split_identifiers(true)
                .preserve_original(true),
            true,
        )
        .await
        .unwrap();
    let query = match_query("targetAlpha", Operator::And);
    assert_eq!(indexed_ids(&dataset, query.clone()).await, [0]);
    assert_eq!(postfilter_ids(&dataset, query).await, [0]);

    let fuzzy_query = FullTextSearchQuery::new_query(FtsQuery::Match(
        MatchQuery::new("targets".to_owned())
            .with_column(Some("docs".to_owned()))
            .with_fuzziness(Some(1)),
    ));
    let query_vector = Float32Array::from(vec![0.0, 0.0]);
    let mut scanner = dataset.scan();
    scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(false)
        .filter_query(QueryFilter::Fts(fuzzy_query))
        .unwrap();
    let error = scanner.try_into_batch().await.unwrap_err();
    assert!(matches!(&error, Error::NotSupported { .. }));
    assert!(
        error
            .to_string()
            .contains("Fuzzy MatchQuery is not supported when FTS is used as a post-filter"),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn test_scan_limit_offset_preserves_json_extension_metadata() {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        json_field("meta", true),
    ]));

    let json_array = JsonArray::try_from_iter((0..50).map(|i| Some(format!(r#"{{"i":{i}}}"#))))
        .unwrap()
        .into_inner();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..50)),
            Arc::new(json_array),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let dataset = Dataset::write(reader, "memory://", None).await.unwrap();

    let mut scanner = dataset.scan();
    scanner.limit(Some(10), None).unwrap();
    let batch_no_offset = scanner.try_into_batch().await.unwrap();
    assert!(is_arrow_json_field(
        batch_no_offset.schema().field_with_name("meta").unwrap()
    ));

    let mut scanner = dataset.scan();
    scanner.limit(Some(10), Some(10)).unwrap();
    let batch_with_offset = scanner.try_into_batch().await.unwrap();
    assert!(is_arrow_json_field(
        batch_with_offset.schema().field_with_name("meta").unwrap()
    ));
    assert_eq!(batch_no_offset.schema(), batch_with_offset.schema());
}

#[tokio::test]
async fn test_scan_nested_arrow_json_extension_v2() {
    let mut json_metadata = HashMap::new();
    json_metadata.insert(
        ARROW_EXT_NAME_KEY.to_string(),
        ARROW_JSON_EXT_NAME.to_string(),
    );
    let item_fields = Fields::from(vec![
        Arc::new(ArrowField::new("uri", DataType::Utf8, false)),
        Arc::new(ArrowField::new("extra", DataType::Utf8, true).with_metadata(json_metadata)),
    ]);
    let item = Arc::new(ArrowField::new(
        "item",
        DataType::Struct(item_fields.clone()),
        true,
    ));
    let media_field = ArrowField::new("media", DataType::List(item.clone()), true);
    let schema = Arc::new(ArrowSchema::new(vec![media_field]));

    for version in [
        LanceFileVersion::V2_1,
        LanceFileVersion::V2_2,
        LanceFileVersion::V2_3,
    ] {
        let values = StructArray::new(
            item_fields.clone(),
            vec![
                Arc::new(StringArray::from(vec![Some("a.jpg"), Some("b.jpg")])) as Arc<dyn Array>,
                Arc::new(StringArray::from(vec![
                    Some(r#"{"codec":"h264"}"#),
                    None::<&str>,
                ])) as Arc<dyn Array>,
            ],
            None,
        );
        let media = ListArray::new(
            item.clone(),
            OffsetBuffer::new(ScalarBuffer::from(vec![0, 1, 2])),
            Arc::new(values),
            None,
        );
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(media)]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema.clone());
        let write_params = WriteParams {
            data_storage_version: Some(version),
            ..WriteParams::default()
        };
        let uri = format!("memory://{}", Uuid::new_v4());
        let dataset = Dataset::write(reader, &uri, Some(write_params))
            .await
            .unwrap();

        let batch = dataset.scan().try_into_batch().await.unwrap();
        let batch_schema = batch.schema();
        let DataType::List(item) = batch_schema.field(0).data_type() else {
            panic!("expected media list field");
        };
        let DataType::Struct(fields) = item.data_type() else {
            panic!("expected media item struct");
        };
        assert!(is_arrow_json_field(&fields[1]));

        let media: &ListArray = batch.column(0).as_list();
        let items = media.values().as_struct();
        let extra = items
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(extra.value(0).contains("h264"));
        assert!(extra.is_null(1));
    }
}

#[tokio::test]
async fn test_scan_miniblock_dictionary_out_of_line_bitpacking_does_not_panic() {
    let rows: usize = 10_000;
    let unique_values: usize = 2_000;
    let batch_size: usize = 8_192;

    let mut field_meta = HashMap::new();
    field_meta.insert(
        "lance-encoding:structural-encoding".to_string(),
        "miniblock".to_string(),
    );
    field_meta.insert(
        "lance-encoding:dict-size-ratio".to_string(),
        "0.99".to_string(),
    );

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("d", DataType::UInt64, false).with_metadata(field_meta),
    ]));

    let values = (0..rows)
        .map(|i| (i % unique_values) as u64)
        .collect::<Vec<_>>();
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(UInt64Array::from(values))]).unwrap();

    let uri = format!("memory://{}", Uuid::new_v4());
    let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema.clone());

    let write_params = WriteParams {
        data_storage_version: Some(LanceFileVersion::V2_2),
        ..WriteParams::default()
    };
    let dataset = Dataset::write(reader, &uri, Some(write_params))
        .await
        .unwrap();

    let field_id = dataset.schema().field("d").unwrap().id as u32;
    let fragment = dataset.get_fragment(0).unwrap();
    let data_file = fragment.data_file_for_field(field_id).unwrap();
    let field_pos = data_file
        .fields
        .iter()
        .position(|id| *id == field_id as i32)
        .unwrap();
    let column_idx = data_file.column_indices[field_pos] as usize;

    let file_path = dataset.data_dir().join(data_file.path.as_str());
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::max_bandwidth(&dataset.object_store),
    );
    let file_scheduler = scheduler
        .open_file(&file_path, &CachedFileSize::unknown())
        .await
        .unwrap();

    let cache = LanceCache::with_capacity(8 * 1024 * 1024);
    let file_reader = FileReader::try_open(
        file_scheduler,
        None,
        Arc::<DecoderPlugins>::default(),
        &cache,
        FileReaderOptions::default(),
    )
    .await
    .unwrap();

    let col_meta = &file_reader.metadata().column_metadatas[column_idx];
    let encoding = describe_encoding(col_meta.pages.first().unwrap());
    assert!(
        encoding.contains("OutOfLineBitpacking") && encoding.contains("dictionary"),
        "Expected a mini-block dictionary page with out-of-line bitpacking, got: {encoding}"
    );

    let mut scanner = dataset.scan();
    scanner.batch_size(batch_size);
    scanner.project(&["d"]).unwrap();

    let mut stream = scanner.try_into_stream().await.unwrap();
    let batch = stream.try_next().await.unwrap().unwrap();
    assert_eq!(batch.num_columns(), 1);
}

async fn prepare_query_filter_dataset() -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                4,
            ),
            true,
        ),
        ArrowField::new("text", DataType::Utf8, false),
        ArrowField::new("category", DataType::Utf8, false),
    ]));

    // Prepare dataset
    let mut vectors = vec![];
    for i in 1..=300 {
        vectors.extend(vec![i as f32; 4]);
    }

    // id 256..298 has noop, others has text
    let mut text = vec![];
    for i in 1..=255 {
        text.push(format!("text {}", i));
    }
    for i in 256..=298 {
        text.push(format!("noop {}", i));
    }
    text.extend(vec!["text 299".to_string(), "text 300".to_string()]);

    let mut category = vec![];
    for i in 1..=300 {
        if i % 3 == 1 {
            category.push("literature".to_string());
        } else if i % 3 == 2 {
            category.push("science".to_string());
        } else {
            category.push("geography".to_string());
        }
    }

    let vectors = Float32Array::from(vectors);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(1..=300)),
            Arc::new(FixedSizeListArray::try_new_from_values(vectors, 4).unwrap()),
            Arc::new(StringArray::from(text)),
            Arc::new(StringArray::from(category)),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

    // Create index
    let params = VectorIndexParams::with_ivf_pq_params(
        MetricType::L2,
        IvfBuildParams::new(2),
        PQBuildParams::new(4, 8),
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default().with_position(true),
            true,
        )
        .await
        .unwrap();

    dataset
}

async fn check_results(
    stream: DatasetRecordBatchStream,
    expected_schema: SchemaRef,
    expected_ids: &[i32],
) {
    let results = stream.try_collect::<Vec<_>>().await.unwrap();
    let batch = concat_batches(&results[0].schema(), &results).unwrap();
    assert_eq!(batch.schema(), expected_schema);

    let ids = batch
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(ids.values(), expected_ids);
}

// ---------------------------------------------------------------------------
// Scanner::candidate_fragments
// ---------------------------------------------------------------------------

/// Four fragments of 100 rows with disjoint `value` ranges (fragment `i` holds
/// `i * 100 .. i * 100 + 100`), optionally indexed on `value`, then a fifth fragment
/// appended *after* indexing so it is genuinely outside the index's coverage.
async fn candidate_fragment_fixture(
    index: Option<IndexType>,
    enable_stable_row_ids: bool,
) -> (Dataset, TempStrDir) {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("value", DataType::Int32, true),
        ArrowField::new("other", DataType::Int32, true),
    ]));
    let batch = |range: std::ops::Range<i32>| {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(range.clone())),
                Arc::new(Int32Array::from_iter_values(range)),
            ],
        )
        .unwrap()
    };

    let test_dir = TempStrDir::default();
    let params = WriteParams {
        max_rows_per_file: 100,
        enable_stable_row_ids,
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(vec![Ok(batch(0..400))], schema.clone());
    let mut dataset = Dataset::write(reader, &test_dir, Some(params.clone()))
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 4);
    // The row-id mode is the axis these tests turn on, so assert the premise rather than
    // trusting the write param: an index answering in the row-id domain can only be
    // distinguished from one answering in the address domain if this actually took effect.
    assert_eq!(
        dataset.manifest.uses_stable_row_ids(),
        enable_stable_row_ids,
        "fixture did not produce the requested row-id mode"
    );

    if let Some(index_type) = index {
        let builtin = match index_type {
            IndexType::BTree => BuiltinIndexType::BTree,
            IndexType::ZoneMap => BuiltinIndexType::ZoneMap,
            other => panic!("fixture does not build {other} indices"),
        };
        dataset
            .create_index(
                &["value"],
                index_type,
                None,
                &ScalarIndexParams::for_builtin(builtin),
                true,
            )
            .await
            .unwrap();
    }

    let append = WriteParams {
        mode: WriteMode::Append,
        ..params
    };
    let reader = RecordBatchIterator::new(vec![Ok(batch(400..500))], schema);
    let dataset = Dataset::write(reader, &test_dir, Some(append))
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 5);
    assert_eq!(
        dataset.manifest.uses_stable_row_ids(),
        enable_stable_row_ids,
        "append must preserve the row-id mode"
    );
    (dataset, test_dir)
}

fn frag_set(ids: impl IntoIterator<Item = u32>) -> RoaringBitmap {
    ids.into_iter().collect()
}

/// The fragments that really hold a row matching `filter`, from an actual scan.
///
/// Pruning may only drop a fragment that holds no match, so this set must always be a
/// subset of [`FragmentCandidates::to_scan`]. Without this cross-check a test could pass
/// while the API silently lost rows.
async fn matching_fragments(dataset: &Dataset, filter: &str) -> RoaringBitmap {
    let batch = dataset
        .scan()
        .with_row_address()
        .filter(filter)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    batch[ROW_ADDR]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .map(|addr| RowAddress::from(*addr).fragment_id())
        .collect()
}

async fn candidates_for(dataset: &Dataset, filter: &str) -> FragmentCandidates {
    let mut scan = dataset.scan();
    scan.filter(filter).unwrap();
    let candidates = scan.candidate_fragments().await.unwrap();
    assert!(
        (&candidates.hits & &candidates.unknown).is_empty(),
        "hits and unknown must be disjoint: {candidates:?}"
    );
    let matching = matching_fragments(dataset, filter).await;
    assert!(
        (&matching - &candidates.to_scan()).is_empty(),
        "pruning lost fragments that hold matching rows: matching={matching:?} \
         candidates={candidates:?}"
    );
    candidates
}

/// With no index nothing can be ruled out, so every fragment is unjudged.
#[tokio::test]
async fn test_candidate_fragments_without_index() {
    let (dataset, _dir) = candidate_fragment_fixture(None, false).await;
    let candidates = candidates_for(&dataset, "value = 150").await;
    assert_eq!(candidates.hits, RoaringBitmap::new());
    assert_eq!(candidates.unknown, frag_set(0..5));
}

/// The core pruning assertion, across the index types and both row-id modes.
///
/// Every index type a current writer produces answers in the row-address domain -- a zone
/// map always, and a BTree at `BTREE_ROW_ADDR_DOMAIN_VERSION` and later -- so pruning
/// works in both row-id modes. The row-id mode is parameterized anyway because it decides
/// whether a mask key coincides with a fragment id by accident (addresses) or has to be
/// kept in the address domain on purpose (stable row ids): a regression that translated
/// results to row ids would still pass the address cases and fail the stable ones.
#[rstest]
#[case::btree_addr(IndexType::BTree, false)]
#[case::btree_stable(IndexType::BTree, true)]
#[case::zonemap_addr(IndexType::ZoneMap, false)]
#[case::zonemap_stable(IndexType::ZoneMap, true)]
#[tokio::test]
async fn test_candidate_fragments_prunes(
    #[case] index_type: IndexType,
    #[case] enable_stable_row_ids: bool,
) {
    let (dataset, _dir) = candidate_fragment_fixture(Some(index_type), enable_stable_row_ids).await;
    let candidates = candidates_for(&dataset, "value = 150").await;

    // Only fragment 1 holds value 150; fragment 4 was appended after indexing.
    assert_eq!(candidates.hits, frag_set([1]));
    assert_eq!(candidates.unknown, frag_set([4]));
    assert_eq!(candidates.to_scan(), frag_set([1, 4]));
}

/// A zone map answers `AtMost`, whose `lower` bound is empty. Pruning on `lower` instead
/// of `upper` would report no candidates at all for a filter that does match.
#[tokio::test]
async fn test_candidate_fragments_inexact_index_uses_upper_bound() {
    let (dataset, _dir) = candidate_fragment_fixture(Some(IndexType::ZoneMap), false).await;
    let candidates = candidates_for(&dataset, "value >= 150 AND value < 250").await;
    assert_eq!(candidates.hits, frag_set([1, 2]));
    assert!(
        !candidates.hits.is_empty(),
        "upper bound must admit matches"
    );
}

/// A fully deleted fragment holds no row, so it belongs to neither set even though the
/// index still covers it and its summaries still claim its old values.
#[tokio::test]
async fn test_candidate_fragments_skips_fully_deleted_fragment() {
    let (mut dataset, _dir) = candidate_fragment_fixture(Some(IndexType::ZoneMap), false).await;
    dataset
        .delete("value >= 100 AND value < 200")
        .await
        .unwrap();

    let candidates = candidates_for(&dataset, "value = 150").await;
    assert!(
        !candidates.to_scan().contains(1),
        "a fully deleted fragment must not be a candidate: {candidates:?}"
    );
    assert_eq!(candidates.hits, RoaringBitmap::new());
    assert_eq!(candidates.unknown, frag_set([4]));
}

/// `NOT` over an exact index yields a block-list `upper`, which may select rows in any
/// fragment and therefore cannot prune.
#[tokio::test]
async fn test_candidate_fragments_block_list_is_not_prunable() {
    let (dataset, _dir) = candidate_fragment_fixture(Some(IndexType::BTree), false).await;
    let candidates = candidates_for(&dataset, "value != 150").await;
    assert_eq!(candidates.hits, frag_set(0..4));
    assert_eq!(candidates.unknown, frag_set([4]));
}

/// An `OR` whose other side has no index cannot be answered by the index at all, so the
/// filter falls back to a refine-only plan and nothing is pruned.
#[tokio::test]
async fn test_candidate_fragments_or_with_unindexed_column() {
    let (dataset, _dir) = candidate_fragment_fixture(Some(IndexType::BTree), false).await;
    let candidates = candidates_for(&dataset, "value = 150 OR other = 250").await;
    assert_eq!(candidates.hits, RoaringBitmap::new());
    assert_eq!(candidates.unknown, frag_set(0..5));
}

/// A predicate part the index could not absorb is ignored for pruning, so a fragment can
/// be reported as a hit while holding no actual match. Documents the over-approximation.
#[tokio::test]
async fn test_candidate_fragments_over_reports_when_refine_remains() {
    let (dataset, _dir) = candidate_fragment_fixture(Some(IndexType::BTree), false).await;
    let filter = "value = 150 AND other = 999";
    assert!(
        matching_fragments(&dataset, filter).await.is_empty(),
        "filter must match nothing for this test to mean anything"
    );
    let candidates = candidates_for(&dataset, filter).await;
    assert_eq!(candidates.hits, frag_set([1]));
}

/// NULL edge cases: a fragment with no nulls must be excluded by `IS NULL`, and an
/// all-null fragment must be admitted.
#[rstest]
#[case::btree(IndexType::BTree)]
#[case::zonemap(IndexType::ZoneMap)]
#[tokio::test]
async fn test_candidate_fragments_is_null(#[case] index_type: IndexType) {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "value",
        DataType::Int32,
        true,
    )]));
    // Fragment 0 has no nulls, fragment 1 is entirely null.
    let values = (0..100)
        .map(Some)
        .chain((0..100).map(|_| None))
        .collect::<Int32Array>();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(values)]).unwrap();
    let test_dir = TempStrDir::default();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut dataset = Dataset::write(
        reader,
        &test_dir,
        Some(WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    let builtin = match index_type {
        IndexType::BTree => BuiltinIndexType::BTree,
        _ => BuiltinIndexType::ZoneMap,
    };
    dataset
        .create_index(
            &["value"],
            index_type,
            None,
            &ScalarIndexParams::for_builtin(builtin),
            true,
        )
        .await
        .unwrap();

    let candidates = candidates_for(&dataset, "value IS NULL").await;
    assert_eq!(candidates.hits, frag_set([1]));
    assert_eq!(candidates.unknown, RoaringBitmap::new());
}

/// Candidate fragments are only defined for a scalar filter.
#[tokio::test]
async fn test_candidate_fragments_rejects_search_queries() {
    let (dataset, _dir) = candidate_fragment_fixture(Some(IndexType::BTree), false).await;
    let mut scan = dataset.scan();
    scan.full_text_search(FullTextSearchQuery::new("hello".to_string()))
        .unwrap();
    let error = scan.candidate_fragments().await.unwrap_err();
    assert!(
        matches!(error, Error::NotSupported { .. }),
        "expected NotSupported, got {error:?}"
    );
    assert!(
        error
            .to_string()
            .contains("only defined for a scalar filter"),
        "{error}"
    );

    let mut scan = dataset.scan();
    scan.minhash_search(MinHashQuery::new("value", "value"))
        .unwrap();
    assert!(matches!(
        scan.candidate_fragments().await,
        Err(Error::NotSupported { .. })
    ));
}

/// The contract of this API: the answer comes from index and manifest data, never from a
/// data file.
///
/// The negative control matters as much as the assertion -- without it, a refactor that
/// made `candidate_fragments` return early would still pass. The fixture asserts the
/// row-id lineage is inline, because a dataset that spills lineage into a data column
/// would read a data file for a legitimate reason and invalidate the premise.
#[rstest]
#[case::addr(false)]
#[case::stable(true)]
#[tokio::test]
async fn test_candidate_fragments_reads_no_data_files(#[case] enable_stable_row_ids: bool) {
    let (dataset, _dir) =
        candidate_fragment_fixture(Some(IndexType::ZoneMap), enable_stable_row_ids).await;
    for fragment in dataset.get_fragments() {
        assert!(
            !fragment.metadata().has_spilled_row_lineage(),
            "fixture must keep row-id lineage out of a data file"
        );
    }
    let data_prefix = format!("/{DATA_DIR}/");

    let _ = dataset.object_store.io_stats_incremental();
    let candidates = dataset
        .scan()
        .filter("value = 150")
        .unwrap()
        .candidate_fragments()
        .await
        .unwrap();
    assert_eq!(candidates.hits, frag_set([1]));
    let pruning_reads = dataset
        .object_store
        .io_stats_incremental()
        .requests
        .into_iter()
        .filter(|request| request.path.as_ref().contains(&data_prefix))
        .collect::<Vec<_>>();
    assert!(
        pruning_reads.is_empty(),
        "candidate_fragments must not read data files: {pruning_reads:#?}"
    );

    // Negative control: the equivalent scan does read data files.
    let _ = dataset.object_store.io_stats_incremental();
    dataset
        .scan()
        .filter("value = 150")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let scan_reads = dataset
        .object_store
        .io_stats_incremental()
        .requests
        .into_iter()
        .filter(|request| request.path.as_ref().contains(&data_prefix))
        .count();
    assert!(
        scan_reads > 0,
        "the control scan should have read data files"
    );
}

/// `with_fragments` narrows the universe: a fragment outside the scope is in neither set,
/// so the index's wider coverage must not leak scope-external fragments into the answer.
#[tokio::test]
async fn test_candidate_fragments_honors_fragment_scope() {
    let (dataset, _dir) = candidate_fragment_fixture(Some(IndexType::ZoneMap), false).await;
    let scoped = dataset
        .get_fragments()
        .into_iter()
        .filter(|fragment| matches!(fragment.id(), 1 | 2))
        .map(|fragment| fragment.metadata().clone())
        .collect::<Vec<_>>();
    assert_eq!(scoped.len(), 2);

    let mut scan = dataset.scan();
    scan.with_fragments(scoped).filter("value = 150").unwrap();
    let candidates = scan.candidate_fragments().await.unwrap();

    assert_eq!(candidates.hits, frag_set([1]));
    assert_eq!(candidates.unknown, RoaringBitmap::new());
    // Fragment 4 is uncovered by the index, but it is also out of scope, so it must not
    // be reported.
    assert!(!candidates.to_scan().contains(4), "{candidates:?}");
}

/// Compaction rewrites both the fragments and the indices' declared coverage, and a zone
/// map cannot be remapped at all. Pruning afterwards must not lose a fragment that holds
/// a match -- the hazard being an index whose stored addresses and whose declared
/// coverage disagree about fragment ids.
///
/// The exact sets are deliberately not asserted: they depend on whether the index type
/// survives compaction. `candidates_for` cross-checks against a real scan, which is the
/// property that actually matters here.
#[rstest]
#[case::btree(IndexType::BTree)]
#[case::zonemap(IndexType::ZoneMap)]
#[tokio::test]
async fn test_candidate_fragments_after_compaction(#[case] index_type: IndexType) {
    let (mut dataset, _dir) = candidate_fragment_fixture(Some(index_type), false).await;
    crate::dataset::optimize::compact_files(&mut dataset, Default::default(), None)
        .await
        .unwrap();

    let candidates = candidates_for(&dataset, "value = 150").await;
    assert!(
        !candidates.to_scan().is_empty(),
        "the matching row must still be reachable: {candidates:?}"
    );
}
