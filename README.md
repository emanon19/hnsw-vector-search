# Rust HNSW Vector Search Engine

A vector search engine written in Rust that supports exact cosine similarity search and approximate nearest-neighbor search using **Hierarchical Navigable Small World (HNSW)** graphs.

## Features

- Vector embeddings with cosine similarity.
- Exact top-k nearest-neighbor search.
- Concurrent exact search using scoped threads.
- HNSW graph construction and approximate nearest-neighbor search.
- Configurable HNSW parameters.
- Vector ID lookup and duplicate ID validation.
- Input validation for dimensions, zero vectors, and non-finite values.
- Benchmarking for search performance and recall.

## Project Structure

```text
vector_engine/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── error.rs
    ├── vector.rs
    ├── index.rs
    └── hnsw.rs
```

## Requirements

- Rust
- Cargo

## Build

From the project root, run:

```bash
cargo check -p vector_engine
```

To compile the tests as well:

```bash
cargo check -p vector_engine --tests
```

Run the available tests with:

```bash
cargo test -p vector_engine
```

## Core Components

### Vector Embeddings

`vector.rs` defines vector IDs and embeddings, validates vector data, and provides cosine similarity calculations.

### Exact Vector Search

`index.rs` implements `VectorIndex`, which supports:

- Inserting vectors.
- Retrieving vectors by ID.
- Searching for the top-k most similar vectors.
- Concurrent exact search.

Exact search is useful as a baseline for evaluating approximate search quality.

### HNSW Search

`hnsw.rs` implements `HnswIndex`, which builds a layered graph to find approximate nearest neighbors without comparing every stored vector for each query.

The main parameters are:

| Parameter | Description |
|---|---|
| `dimension` | Number of values in each vector |
| `m` | Maximum neighbor count used when building graph layers |
| `ef_construction` | Candidate search width during graph construction |
| `ef_search` | Candidate search width during queries |

Increasing `ef_search` can improve recall but may also increase query latency.

## Benchmarking

The benchmark compares HNSW search against exact search.

Previously measured configuration:

- Dataset: 10,000 vectors.
- Dimensions: 128.
- Queries: 100.
- Results per query: top 10.
- Recall@10 at `ef = 256`: approximately 98.1%.
- Average HNSW query latency at `ef = 200`: approximately 1.09 ms.

These figures are results from a previous run, not guaranteed performance. Actual results depend on hardware, compiler settings, dataset characteristics, and benchmark implementation.

Run the existing ignored benchmark in release mode with:

```bash
cargo test --release -p vector_engine benchmark_hnsw -- --ignored --nocapture
```

## Current Scope

This project focuses on the core vector search engine:

- Vector storage and similarity calculations.
- Exact and concurrent search.
- HNSW indexing and approximate search.
- Benchmarking recall and latency.

It does not require a web API, external database, or deployment infrastructure.

## License

MIT License
