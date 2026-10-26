
use std::collections::HashSet;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use instant_distance::{Builder, HnswMap, Point, Search};
use vector_engine::hnsw::HnswIndex;
use vector_engine::index::{SearchResult, VectorIndex};
use vector_engine::vector::VectorId;

const DIMENSION: usize = 128;
const VECTOR_COUNT: usize = 10_000;
const TOP_K: usize = 10;
const EVALUATION_QUERY_COUNT: usize = 30;
const BUILD_COUNT: usize = 500;
const EF_SEARCH_VALUES: [usize; 3] = [16, 64, 128];
const EF_CONSTRUCTION: usize = 100;
const BASELINE_EF_SEARCH: usize = 64;

#[derive(Clone)]
struct CosinePoint {
    data: Vec<f32>,
    norm: f32,
}

impl CosinePoint {
    fn new(data: Vec<f32>) -> Self {
        let norm = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        Self { data, norm }
    }
}

impl Point for CosinePoint {
    fn distance(&self, other: &Self) -> f32 {
        let dot = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .sum::<f32>();

        if self.norm == 0.0 || other.norm == 0.0 {
            return 1.0;
        }

        1.0 - (dot / (self.norm * other.norm)).clamp(-1.0, 1.0)
    }
}

type BaselineIndex = HnswMap<CosinePoint, u64>;

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

fn make_query(query_index: usize, dimension: usize) -> Vec<f32> {
    let base_seed = 50_000_000_u64
        .wrapping_add((query_index as u64).wrapping_mul(dimension as u64));

    (0..dimension)
        .map(|j| sample(base_seed.wrapping_add(j as u64)))
        .collect()
}

fn build_exact(count: usize, dimension: usize) -> VectorIndex {
    let mut index = VectorIndex::new(dimension).unwrap();

    for id in 0..count {
        index
            .insert(VectorId(id as u64), vector_for(id, dimension))
            .unwrap();
    }

    index
}

fn build_hnsw(count: usize, dimension: usize) -> HnswIndex {
    let mut index =
        HnswIndex::new(dimension, 16, EF_CONSTRUCTION, BASELINE_EF_SEARCH).unwrap();

    for id in 0..count {
        index
            .insert(VectorId(id as u64), vector_for(id, dimension))
            .unwrap();
    }

    index
}

fn build_baseline(count: usize, dimension: usize) -> BaselineIndex {
    let points: Vec<CosinePoint> = (0..count)
        .map(|id| CosinePoint::new(vector_for(id, dimension)))
        .collect();

    let values: Vec<u64> = (0..count as u64).collect();

    Builder::default()
        .ef_construction(EF_CONSTRUCTION)
        .ef_search(BASELINE_EF_SEARCH)
        .seed(42)
        .build(points, values)
}

fn recall_at_k(exact: &[SearchResult], approximate_ids: &[u64]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }

    let exact_ids: HashSet<u64> =
        exact.iter().map(|result| result.id.0).collect();

    let hits = approximate_ids
        .iter()
        .filter(|id| exact_ids.contains(id))
        .count();

    hits as f64 / exact.len() as f64
}

fn evaluate_recall(
    exact: &VectorIndex,
    hnsw: &HnswIndex,
    baseline: &BaselineIndex,
) {
    println!(
        "\nRecall@{TOP_K}: {VECTOR_COUNT} vectors, \
         {DIMENSION} dimensions, {EVALUATION_QUERY_COUNT} queries"
    );
    println!("{:<28} {:>12}", "Index / ef_search", "Recall");

    let mut own_hits = [0usize; 3];
    let mut baseline_hits = 0usize;
    let mut expected_total = 0usize;

    let mut search = Search::default();

    for query_index in 0..EVALUATION_QUERY_COUNT {
        let raw_query = make_query(query_index, DIMENSION);
        let query_point = CosinePoint::new(raw_query.clone());
        let truth = exact.search(&raw_query, TOP_K).unwrap();

        expected_total += truth.len();

        for (i, ef) in EF_SEARCH_VALUES.iter().enumerate() {
            let found = hnsw
                .search_with_ef(&raw_query, TOP_K, *ef)
                .unwrap();

            let found_ids: Vec<u64> =
                found.iter().map(|result| result.id.0).collect();

            let truth_ids: HashSet<u64> =
                truth.iter().map(|result| result.id.0).collect();

            own_hits[i] += found_ids
                .iter()
                .filter(|id| truth_ids.contains(id))
                .count();
        }

        let baseline_ids: Vec<u64> = baseline
            .search(&query_point, &mut search)
            .take(TOP_K)
            .map(|item| *item.value)
            .collect();

        let truth_ids: HashSet<u64> =
            truth.iter().map(|result| result.id.0).collect();

        baseline_hits += baseline_ids
            .iter()
            .filter(|id| truth_ids.contains(id))
            .count();
    }

    for (i, ef) in EF_SEARCH_VALUES.iter().enumerate() {
        println!(
            "{:<28} {:>11.2}%",
            format!("vector_engine ef={ef}"),
            own_hits[i] as f64 / expected_total as f64 * 100.0
        );
    }

    println!(
        "{:<28} {:>11.2}%",
        format!("instant-distance ef={BASELINE_EF_SEARCH}"),
        baseline_hits as f64 / expected_total as f64 * 100.0
    );
}

fn benchmark_search(c: &mut Criterion) {
    eprintln!("Building exact index...");
    let exact = build_exact(VECTOR_COUNT, DIMENSION);

    eprintln!("Building vector_engine HNSW...");
    let hnsw = build_hnsw(VECTOR_COUNT, DIMENSION);

    eprintln!("Building instant-distance baseline...");
    let baseline = build_baseline(VECTOR_COUNT, DIMENSION);

    evaluate_recall(&exact, &hnsw, &baseline);

    let raw_query = make_query(0, DIMENSION);
    let query_point = CosinePoint::new(raw_query.clone());

    let mut group = c.benchmark_group("search_100k_128d");
    group.sample_size(10);

    group.bench_function("exact_single_thread_top10", |b| {
        b.iter(|| exact.search(black_box(&raw_query), TOP_K).unwrap())
    });

    for ef in EF_SEARCH_VALUES {
        group.bench_function(format!("vector_engine_ef{ef}_top10"), |b| {
            b.iter(|| {
                hnsw.search_with_ef(black_box(&raw_query), TOP_K, ef)
                    .unwrap()
            })
        });
    }

    let mut baseline_search = Search::default();
    group.bench_function(
        format!("instant_distance_ef{BASELINE_EF_SEARCH}_top10"),
        |b| {
            b.iter(|| {
                baseline
                    .search(black_box(&query_point), &mut baseline_search)
                    .take(TOP_K)
                    .count()
            })
        },
    );

    group.finish();
}

fn benchmark_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_500_128d");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("exact", |b| {
        b.iter(|| black_box(build_exact(BUILD_COUNT, DIMENSION)))
    });

    group.bench_function("vector_engine_hnsw", |b| {
        b.iter(|| black_box(build_hnsw(BUILD_COUNT, DIMENSION)))
    });

    group.bench_function("instant_distance", |b| {
        b.iter(|| black_box(build_baseline(BUILD_COUNT, DIMENSION)))
    });

    group.finish();
}

criterion_group!(benches, benchmark_search, benchmark_build);
criterion_main!(benches);
