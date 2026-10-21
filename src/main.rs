use std::time::Instant;

use vector_engine::index::VectorIndex;
use vector_engine::vector::VectorId;

// SplitMix64: a small deterministic generator suitable for test data.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

// Generate a reproducible f32 value approximately in [-1, 1).
fn sample(seed: u64) -> f32 {
    let bits = splitmix64(seed) >> 40;
    (bits as f32 / 16_777_216.0) * 2.0 - 1.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dimension = 128;
    let vector_count = 100_000;

    println!("Initializing Vector Engine...");

    let mut index = VectorIndex::new(dimension)?;
    let start = Instant::now();

    for i in 0..vector_count {
        let data: Vec<f32> = (0..dimension)
            .map(|j| {
                sample(
                    (i as u64)
                        .wrapping_mul(dimension as u64)
                        .wrapping_add(j as u64),
                )
            })
            .collect();

        index.insert(VectorId(i as u64), data)?;
    }

    println!("Inserted {} vectors in {:?}", index.len(), start.elapsed());

    // Use a separate seed range for the query.
    let query: Vec<f32> = (0..dimension)
        .map(|j| sample(10_000_000 + j as u64))
        .collect();

    let start = Instant::now();
    let results = index.search(&query, 5)?;

    println!("Search completed in {:?}", start.elapsed());

    for result in results {
        println!(
            "ID: {}, cosine similarity: {:.8}",
            result.id.0, result.similarity
        );
    }

    Ok(())
}
