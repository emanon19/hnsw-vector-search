use vector_engine::error::SearchError;
use vector_engine::hnsw::HnswIndex;
use vector_engine::vector::VectorId;

fn sample_hnsw() -> HnswIndex {
    let mut index = HnswIndex::new(3, 4, 16, 16).unwrap();

    index.insert(VectorId(30), vec![0.0, 1.0, 0.0]).unwrap();
    index.insert(VectorId(20), vec![1.0, 0.0, 0.0]).unwrap();
    index.insert(VectorId(10), vec![1.0, 0.0, 0.0]).unwrap();
    index.insert(VectorId(40), vec![-1.0, 0.0, 0.0]).unwrap();
    index.insert(VectorId(50), vec![0.0, 0.0, 1.0]).unwrap();

    index
}

#[test]
fn hnsw_returns_nearest_neighbors_in_rank_order() {
    let index = sample_hnsw();
    let results = index.search(&[1.0, 0.0, 0.0], 3).unwrap();

    let ids: Vec<u64> = results.iter().map(|r| r.id.0).collect();

    assert_eq!(ids, vec![10, 20, 30]);
}

#[test]
fn hnsw_rejects_duplicate_ids() {
    let mut index = sample_hnsw();

    let result = index.insert(VectorId(10), vec![0.0, 0.0, 1.0]);

    assert!(matches!(result, Err(SearchError::DuplicateId(10))));
}

#[test]
fn hnsw_rejects_invalid_configuration() {
    assert!(matches!(
        HnswIndex::new(3, 1, 16, 16),
        Err(SearchError::InvalidHnswConfig)
    ));

    assert!(matches!(
        HnswIndex::new(3, 4, 2, 16),
        Err(SearchError::InvalidHnswConfig)
    ));
}

#[test]
fn hnsw_search_with_larger_ef_returns_requested_count() {
    let index = sample_hnsw();

    let results = index
        .search_with_ef(&[1.0, 0.0, 0.0], 5, 32)
        .unwrap();

    assert_eq!(results.len(), 5);
}

#[test]
fn hnsw_handles_zero_top_k() {
    let index = sample_hnsw();

    assert!(index.search(&[1.0, 0.0, 0.0], 0).unwrap().is_empty());
}