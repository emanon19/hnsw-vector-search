use vector_engine::error::SearchError;
use vector_engine::index::VectorIndex;
use vector_engine::vector::VectorId;

#[test]
fn returns_most_similar_vector_first() {
    let mut index = VectorIndex::new(2).unwrap();

    index.insert(VectorId(1), vec![1.0, 0.0]).unwrap();

    index.insert(VectorId(2), vec![0.0, 1.0]).unwrap();

    let results = index.search(&[1.0, 0.0], 1).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, VectorId(1));
    assert!((results[0].similarity - 1.0).abs() < 1e-6);
}

#[test]
fn rejects_duplicate_ids() {
    let mut index = VectorIndex::new(2).unwrap();

    index.insert(VectorId(1), vec![1.0, 0.0]).unwrap();

    let result = index.insert(VectorId(1), vec![0.0, 1.0]);

    assert!(matches!(result, Err(SearchError::DuplicateId(1))));
}

#[test]
fn rejects_dimension_mismatch() {
    let mut index = VectorIndex::new(2).unwrap();

    let result = index.insert(VectorId(1), vec![1.0, 0.0, 0.0]);

    assert!(matches!(result, Err(SearchError::DimensionMismatch { .. })));
}

#[test]
fn returns_empty_for_zero_top_k() {
    let mut index = VectorIndex::new(2).unwrap();

    index.insert(VectorId(1), vec![1.0, 0.0]).unwrap();

    assert!(index.search(&[1.0, 0.0], 0).unwrap().is_empty());
}

#[test]
fn rejects_zero_query() {
    let mut index = VectorIndex::new(2).unwrap();

    index.insert(VectorId(1), vec![1.0, 0.0]).unwrap();

    assert!(matches!(
        index.search(&[0.0, 0.0], 1),
        Err(SearchError::ZeroNorm)
    ));
}

fn sample_index() -> VectorIndex {
    let mut index = VectorIndex::new(3).unwrap();

    index.insert(VectorId(30), vec![0.0, 1.0, 0.0]).unwrap();
    index.insert(VectorId(20), vec![1.0, 0.0, 0.0]).unwrap();
    index.insert(VectorId(10), vec![1.0, 0.0, 0.0]).unwrap();
    index.insert(VectorId(40), vec![-1.0, 0.0, 0.0]).unwrap();

    index
}

#[test]
fn heap_search_returns_best_results_in_deterministic_order() {
    let index = sample_index();
    let results = index.search(&[1.0, 0.0, 0.0], 3).unwrap();

    assert_eq!(
        results.iter().map(|r| r.id.0).collect::<Vec<_>>(),
        vec![10, 20, 30]
    );
}

#[test]
fn concurrent_search_matches_sequential_search() {
    let index = sample_index();
    let query = [1.0, 0.0, 0.0];

    let sequential = index.search(&query, 4).unwrap();
    let concurrent = index.search_concurrent(&query, 4, 2).unwrap();

    assert_eq!(sequential, concurrent);
}

#[test]
fn concurrent_search_rejects_zero_workers() {
    let index = sample_index();

    assert!(index.search_concurrent(&[1.0, 0.0, 0.0], 2, 0).is_err());
}

#[test]
fn concurrent_search_handles_more_workers_than_vectors() {
    let index = sample_index();
    let results = index.search_concurrent(&[1.0, 0.0, 0.0], 2, 32).unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id.0, 10);
    assert_eq!(results[1].id.0, 20);
}

fn build_test_index() -> VectorIndex {
    let mut index = VectorIndex::new(3).unwrap();

    index.insert(VectorId(30), vec![0.0, 1.0, 0.0]).unwrap();
    index.insert(VectorId(20), vec![1.0, 0.0, 0.0]).unwrap();
    index.insert(VectorId(10), vec![1.0, 0.0, 0.0]).unwrap();
    index.insert(VectorId(40), vec![-1.0, 0.0, 0.0]).unwrap();
    index.insert(VectorId(50), vec![0.0, 0.0, 1.0]).unwrap();

    index
}

#[test]
fn heap_search_matches_expected_order_with_tied_scores() {
    let index = build_test_index();

    let results = index.search(&[1.0, 0.0, 0.0], 5).unwrap();

    let ids: Vec<u64> = results.iter().map(|r| r.id.0).collect();

    // IDs 10 and 20 have identical vectors and scores.
    // Lower IDs must rank first when scores tie.
    assert_eq!(ids, vec![10, 20, 30, 50, 40]);
}

#[test]
fn heap_top_k_matches_full_sort_reference() {
    let dimension = 16;
    let count = 256;
    let mut index = VectorIndex::new(dimension).unwrap();

    // Deterministic, varied vectors without an additional RNG dependency.
    for i in 0..count {
        let vector: Vec<f32> = (0..dimension)
            .map(|j| {
                let x = (i * dimension + j) as f32;
                (x * 0.173).sin() + (x * 0.071).cos()
            })
            .collect();

        index.insert(VectorId(i as u64), vector).unwrap();
    }

    let query: Vec<f32> = (0..dimension)
        .map(|j| ((j as f32) * 0.137).cos() + 0.2)
        .collect();

    // Independent full-sort reference implementation.
    let query_norm = query
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();

    let mut reference: Vec<(u64, f32)> = (0..count)
        .map(|i| {
            let vector: Vec<f32> = (0..dimension)
                .map(|j| {
                    let x = (i * dimension + j) as f32;
                    (x * 0.173).sin() + (x * 0.071).cos()
                })
                .collect();

            let norm = vector
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                .sqrt();

            let dot = vector
                .iter()
                .zip(&query)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>();

            (i as u64, (dot / (norm * query_norm)) as f32)
        })
        .collect();

    reference.sort_unstable_by(|a, b| {
        b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
    });

    for k in [1, 5, 10, 32, count] {
        let actual = index.search(&query, k).unwrap();

        let expected_ids: Vec<u64> = reference
            .iter()
            .take(k)
            .map(|(id, _)| *id)
            .collect();

        let actual_ids: Vec<u64> = actual.iter().map(|r| r.id.0).collect();

        assert_eq!(actual_ids, expected_ids, "top-k mismatch for k={k}");

        for (result, (_, expected_score)) in actual.iter().zip(reference.iter()) {
            assert!(
                (result.similarity - expected_score).abs() < 1e-6,
                "score mismatch for ID {}",
                result.id.0
            );
        }
    }
}

#[test]
fn concurrent_search_matches_heap_search() {
    let index = build_test_index();
    let query = [1.0, 0.0, 0.0];

    for k in [1, 2, 4, 5, 10] {
        let sequential = index.search(&query, k).unwrap();
        let concurrent = index.search_concurrent(&query, k, 3).unwrap();

        assert_eq!(
            sequential, concurrent,
            "concurrent search mismatch for k={k}"
        );
    }
}
