use std::time::Instant;
use vector_engine::hnsw::HnswIndex;
use vector_engine::vector::VectorId;

const DIMENSION: usize = 128;
const VECTOR_COUNT: usize = 10_000;

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

fn main() {
    let vectors: Vec<Vec<f32>> = (0..VECTOR_COUNT)
        .map(|id| {
            (0..DIMENSION)
                .map(|j| {
                    sample(
                        (id as u64)
                            .wrapping_mul(DIMENSION as u64)
                            .wrapping_add(j as u64),
                    )
                })
                .collect()
        })
        .collect();

    let start = Instant::now();

    let mut index = HnswIndex::new(DIMENSION, 16, 100, 64).unwrap();

    for (id, vector) in vectors.into_iter().enumerate() {
        index.insert(VectorId(id as u64), vector).unwrap();
    }

    println!(
        "Indexed {VECTOR_COUNT} vectors in {:.3}s",
        start.elapsed().as_secs_f64()
    );

    // Keep the constructed index alive through the timing measurement.
    std::hint::black_box(&index);
}