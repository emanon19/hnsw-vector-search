use std::collections::HashSet;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vector_engine::hnsw::HnswIndex;
use vector_engine::index::VectorIndex;
use vector_engine::vector::VectorId;

const DIMENSION: usize = 128;
const VECTOR_COUNT: usize = 100_000;
const TOP_K: usize = 10;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn sample(seed: u64) -> f32 {
    let bits = splitmix64(seed) >> 40;
    (bits as f32 / 16_777_216.0) * 2.0 - 1.0
}

fn vector_for(id: usize, dimension: usize) -> Vec<f32> {
    (0..dimension)
        .map(|j| {
            sample(
                (id as u64)
                    .wrapping_mul(dimension as u64)
                    .wrapping_add(j as u64),
            )
        })
        .collect()
}

fn build_exact(count: usize, dimension: usize) -> VectorIndex {
    let mut index = VectorIndex::new(dimension).unwrap();

    for id in 0..count {
        index
            .insert(
                VectorId(id as u64),
                vector_for(id, dimension),
            )
            .unwrap();
    }

    index
}

fn build_hnsw(count: usize, dimension: usize) -> HnswIndex {
    let mut index = HnswIndex::new(dimension, 16, 100, 64).unwrap();

    for id in 0..count {
        index
            .insert(
                VectorId(id as u64),
                vector_for(id, dimension),
            )
            .unwrap();
    }

    index
}

fn make_query(dimension: usize) -> Vec<f32> {
    (0..dimension)
        .map(|j| sample(50_000_000 + j as u64))
        .collect()
}

fn recall_at_k(
    exact: &[vector_engine::index::SearchResult],
    approximate: &[vector_engine::index::SearchResult],
) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }

    let exact_ids: HashSet<_> = exact.iter().map(|result| result.id).collect();

    let hits = approximate
        .iter()
        .filter(|result| exact_ids.contains(&result.id))
        .count();

    hits as f64 / exact.len() as f64
}

fn benchmark_search(c: &mut Criterion) {
    let query = make_query(DIMENSION);

    eprintln!("Building exact index with {VECTOR_COUNT} vectors...");
    let exact = build_exact(VECTOR_COUNT, DIMENSION);

    eprintln!("Building HNSW index with {VECTOR_COUNT} vectors...");
    let hnsw = build_hnsw(VECTOR_COUNT, DIMENSION);

    let exact_results = exact.search(&query, TOP_K).unwrap();

    eprintln!("\nRecall@{TOP_K} against exact search:");
    for ef in [16, 64, 128] {
        let approximate = hnsw
            .search_with_ef(&query, TOP_K, ef)
            .unwrap();

        eprintln!(
            "ef_search={ef:>3}: recall@{TOP_K} = {:.1}%",
            recall_at_k(&exact_results, &approximate) * 100.0
        );
    }

    let mut group = c.benchmark_group("search_100k_128d");
    group.sample_size(10);

    group.bench_function("exact_single_thread_top10", |b| {
        b.iter(|| exact.search(black_box(&query), TOP_K).unwrap())
    });

    group.bench_function("exact_four_workers_top10", |b| {
        b.iter(|| {
            exact
                .search_concurrent(black_box(&query), TOP_K, 4)
                .unwrap()
        })
    });

    for ef in [16, 64, 128] {
        group.bench_function(format!("hnsw_top10_ef{ef}"), |b| {
            b.iter(|| {
                hnsw.search_with_ef(black_box(&query), TOP_K, ef).unwrap()
            })
        });
    }

    group.finish();
}

fn benchmark_build(c: &mut Criterion) {
    const BUILD_COUNT: usize = 10_000;

    let mut group = c.benchmark_group("build_10k_128d");
    group.sample_size(10);

    group.bench_function("exact", |b| {
        b.iter(|| black_box(build_exact(BUILD_COUNT, DIMENSION)))
    });

    group.bench_function("hnsw_m16_efc100", |b| {
        b.iter(|| black_box(build_hnsw(BUILD_COUNT, DIMENSION)))
    });

    group.finish();
}

criterion_group!(benches, benchmark_search, benchmark_build);
criterion_main!(benches);