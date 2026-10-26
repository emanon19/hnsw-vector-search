use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

use crate::error::SearchError;
use crate::index::SearchResult;
use crate::vector::{Embedding, VectorId};

const MAX_LEVEL: usize = 16;

// -----------------------------------------------------------------------------
// Profiling
// -----------------------------------------------------------------------------

static SEARCH_LAYER_NS: AtomicU64 = AtomicU64::new(0);
static SELECT_NEIGHBORS_NS: AtomicU64 = AtomicU64::new(0);
static PRUNE_NEIGHBORS_NS: AtomicU64 = AtomicU64::new(0);

static SEARCH_LAYER_CALLS: AtomicU64 = AtomicU64::new(0);
static SELECT_NEIGHBORS_CALLS: AtomicU64 = AtomicU64::new(0);
static PRUNE_NEIGHBORS_CALLS: AtomicU64 = AtomicU64::new(0);

struct ProfileTimer {
    nanos: &'static AtomicU64,
    calls: &'static AtomicU64,
    start: Instant,
}

impl ProfileTimer {
    #[inline]
    fn new(
        nanos: &'static AtomicU64,
        calls: &'static AtomicU64,
    ) -> Self {
        Self {
            nanos,
            calls,
            start: Instant::now(),
        }
    }
}

impl Drop for ProfileTimer {
    #[inline]
    fn drop(&mut self) {
        let elapsed_ns = self.start.elapsed().as_nanos() as u64;

        self.nanos
            .fetch_add(elapsed_ns, AtomicOrdering::Relaxed);

        self.calls.fetch_add(1, AtomicOrdering::Relaxed);
    }
}

fn reset_profile() {
    SEARCH_LAYER_NS.store(0, AtomicOrdering::Relaxed);
    SELECT_NEIGHBORS_NS.store(0, AtomicOrdering::Relaxed);
    PRUNE_NEIGHBORS_NS.store(0, AtomicOrdering::Relaxed);

    SEARCH_LAYER_CALLS.store(0, AtomicOrdering::Relaxed);
    SELECT_NEIGHBORS_CALLS.store(0, AtomicOrdering::Relaxed);
    PRUNE_NEIGHBORS_CALLS.store(0, AtomicOrdering::Relaxed);
}

fn print_profile(
    name: &str,
    nanos: &AtomicU64,
    calls: &AtomicU64,
) {
    let total_ms =
        nanos.load(AtomicOrdering::Relaxed) as f64 / 1_000_000.0;
    let count = calls.load(AtomicOrdering::Relaxed);

    let avg_us = if count == 0 {
        0.0
    } else {
        total_ms * 1000.0 / count as f64
    };

    println!(
        "{name:<20} calls={count:>8} total={total_ms:>10.2} ms avg={avg_us:>10.2} us"
    );
}

fn print_profile_report() {
    println!();
    println!("HNSW build profiling:");
    println!(
        "{:<20} {:>14} {:>14} {:>14}",
        "Function", "Calls", "Total (ms)", "Average (us)"
    );
    println!("{}", "-".repeat(66));

    print_profile(
        "search_layer",
        &SEARCH_LAYER_NS,
        &SEARCH_LAYER_CALLS,
    );

    print_profile(
        "select_neighbors",
        &SELECT_NEIGHBORS_NS,
        &SELECT_NEIGHBORS_CALLS,
    );

    print_profile(
        "prune_neighbors",
        &PRUNE_NEIGHBORS_NS,
        &PRUNE_NEIGHBORS_CALLS,
    );
}

// -----------------------------------------------------------------------------
// HNSW candidates
// -----------------------------------------------------------------------------


#[derive(Default)]
struct PruneWorkspace {
    candidates: Vec<Candidate>,
    selected: Vec<usize>,
    selected_marks: Vec<u32>,
    generation: u32,
}

impl PruneWorkspace {
    #[inline]
    fn prepare(&mut self, candidate_capacity: usize, limit: usize) {
        self.candidates.clear();
        self.selected.clear();

        if self.candidates.capacity() < candidate_capacity {
            self.candidates.reserve(candidate_capacity);
        }

        if self.selected.capacity() < limit {
            self.selected.reserve(limit);
        }
    }
}

#[derive(Default)]
struct SearchWorkspace {
    visited_marks: Vec<u32>,
    generation: u32,
}

impl SearchWorkspace {
    #[inline]
    fn begin(&mut self, len: usize) -> u32 {
        if self.visited_marks.len() < len {
            self.visited_marks.resize(len, 0);
        }

        self.generation = self.generation.wrapping_add(1);

        if self.generation == 0 {
            self.visited_marks.fill(0);
            self.generation = 1;
        }

        self.generation
    }

    #[inline]
    fn visit(&mut self, slot: usize, generation: u32) -> bool {
        if self.visited_marks[slot] == generation {
            false
        } else {
            self.visited_marks[slot] = generation;
            true
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    slot: usize,
    id: VectorId,
    similarity: f32,
}

impl Candidate {
    #[inline]
    fn is_better_than(&self, other: &Self) -> bool {
        self.similarity.total_cmp(&other.similarity) == Ordering::Greater
            || (self.similarity.total_cmp(&other.similarity) == Ordering::Equal
                && self.id < other.id)
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.slot == other.slot
            && self.similarity.to_bits() == other.similarity.to_bits()
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.similarity
            .total_cmp(&other.similarity)
            .then_with(|| other.id.cmp(&self.id))
            .then_with(|| self.slot.cmp(&other.slot))
    }
}

// -----------------------------------------------------------------------------
// HNSW index
// -----------------------------------------------------------------------------

pub struct HnswIndex {
    dimension: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    vectors: Vec<Embedding>,
    id_to_slot: HashMap<VectorId, usize>,
    levels: Vec<usize>,
    graph: Vec<Vec<Vec<usize>>>,
    entry_point: Option<usize>,
    max_level: usize,
}

impl HnswIndex {
    pub fn new(
        dimension: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> Result<Self, SearchError> {
        if dimension == 0 || m < 2 || ef_construction < m || ef_search == 0 {
            return Err(SearchError::InvalidHnswConfig);
        }

        Ok(Self {
            dimension,
            m,
            ef_construction,
            ef_search,
            vectors: Vec::new(),
            id_to_slot: HashMap::new(),
            levels: Vec::new(),
            graph: Vec::new(),
            entry_point: None,
            max_level: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn insert(
        &mut self,
        id: VectorId,
        data: Vec<f32>,
    ) -> Result<(), SearchError> {
        if data.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                actual: data.len(),
            });
        }

        if self.id_to_slot.contains_key(&id) {
            return Err(SearchError::DuplicateId(id.0));
        }

        let embedding = Embedding::new(id, data)?;
        let level = random_level(id, self.m);
        let slot = self.vectors.len();

        self.vectors.push(embedding);
        self.levels.push(level);
        self.graph.push(vec![Vec::new(); level + 1]);
        self.id_to_slot.insert(id, slot);

        let Some(mut entry) = self.entry_point else {
            self.entry_point = Some(slot);
            self.max_level = level;
            return Ok(());
        };

        // Own a copy so subsequent graph mutations do not conflict
        // with a borrow into self.vectors.
        let query = self.vectors[slot].clone();
        let old_max_level = self.max_level;

        for layer in ((level + 1)..=old_max_level).rev() {
            entry = self.greedy_search(&query, entry, layer);
        }

        let mut prune_workspace = PruneWorkspace::default();
        let mut search_workspace = SearchWorkspace::default();

        for layer in (0..=level.min(old_max_level)).rev() {
            let candidates = self.search_layer(
                &query,
                &[entry],
                self.ef_construction,
                layer,
                &mut search_workspace,
            );
            
            let next_entry = candidates.first().map(|candidate| candidate.slot);

            let neighbors =
                self.select_neighbors(slot, candidates, self.m_for_layer(layer));

            for &neighbor in &neighbors {
                self.graph[slot][layer].push(neighbor);
                self.graph[neighbor][layer].push(slot);
            }

            // Reuse the same scratch buffers for every prune call.
            for &neighbor in &neighbors {
                self.prune_neighbors(neighbor, layer, &mut prune_workspace);
            }

            if let Some(closest) = next_entry {
                entry = closest;
            }
        }

        if level > old_max_level {
            self.entry_point = Some(slot);
            self.max_level = level;
        }

        Ok(())
    }

    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        self.search_with_ef(query, top_k, self.ef_search)
    }

    pub fn search_with_ef(
        &self,
        query: &[f32],
        top_k: usize,
        ef: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        if top_k == 0 || self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let query = Embedding::new(VectorId(u64::MAX), query.to_vec())?;

        let Some(mut entry) = self.entry_point else {
            return Ok(Vec::new());
        };

        for layer in (1..=self.max_level).rev() {
            entry = self.greedy_search(&query, entry, layer);
        }

        let mut workspace = SearchWorkspace::default();

        let candidates = self.search_layer(
            &query,
            &[entry],
            ef.max(top_k).max(1),
            0,
            &mut workspace,
        );

        Ok(candidates
            .into_iter()
            .take(top_k)
            .map(|candidate| SearchResult {
                id: candidate.id,
                similarity: candidate.similarity,
            })
            .collect())
    }

    fn m_for_layer(&self, layer: usize) -> usize {
        if layer == 0 {
            self.m.saturating_mul(2)
        } else {
            self.m
        }
    }

    #[inline]
    fn score(&self, query: &Embedding, slot: usize) -> f32 {
        query.normalized_dot(&self.vectors[slot])
    }

    #[inline]
    fn candidate(&self, query: &Embedding, slot: usize) -> Candidate {
        let embedding = &self.vectors[slot];

        Candidate {
            slot,
            id: embedding.id,
            similarity: self.score(query, slot),
        }
    }

    fn greedy_search(
        &self,
        query: &Embedding,
        mut current: usize,
        layer: usize,
    ) -> usize {
        let mut current_score = self.score(query, current);

        loop {
            let mut best = current;
            let mut best_score = current_score;

            if layer < self.graph[current].len() {
                for &neighbor in &self.graph[current][layer] {
                    let score = self.score(query, neighbor);

                    if score.total_cmp(&best_score) == Ordering::Greater
                        || (score.total_cmp(&best_score) == Ordering::Equal
                            && self.vectors[neighbor].id
                                < self.vectors[best].id)
                    {
                        best = neighbor;
                        best_score = score;
                    }
                }
            }

            if best == current {
                break;
            }

            current = best;
            current_score = best_score;
        }

        current
    }

    
    fn search_layer(
        &self,
        query: &Embedding,
        entries: &[usize],
        ef: usize,
        layer: usize,
        workspace: &mut SearchWorkspace,
    ) -> Vec<Candidate> {
        let _profile = ProfileTimer::new(
            &SEARCH_LAYER_NS,
            &SEARCH_LAYER_CALLS,
        );

        let ef = ef.max(1);
        let generation = workspace.begin(self.vectors.len());

        let mut exploration = BinaryHeap::<Candidate>::new();
        let mut results = BinaryHeap::<Reverse<Candidate>>::new();

        for &slot in entries {
            if !workspace.visit(slot, generation) {
                continue;
            }

            let candidate = self.candidate(query, slot);
            exploration.push(candidate);
            results.push(Reverse(candidate));
        }

        while let Some(current) = exploration.pop() {
            let Some(Reverse(worst)) = results.peek().copied() else {
                break;
            };

            if results.len() >= ef && !current.is_better_than(&worst) {
                break;
            }

            if layer >= self.graph[current.slot].len() {
                continue;
            }

            for &neighbor in &self.graph[current.slot][layer] {
                if !workspace.visit(neighbor, generation) {
                    continue;
                }

                let candidate = self.candidate(query, neighbor);

                let should_add = results.len() < ef
                    || results
                        .peek()
                        .is_some_and(|Reverse(worst)| {
                            candidate.is_better_than(worst)
                        });

                if !should_add {
                    continue;
                }

                exploration.push(candidate);
                results.push(Reverse(candidate));

                if results.len() > ef {
                    results.pop();
                }
            }
        }

        let mut output: Vec<Candidate> =
            results.into_iter().map(|Reverse(c)| c).collect();

        output.sort_unstable_by(|a, b| {
            b.similarity
                .total_cmp(&a.similarity)
                .then_with(|| a.id.cmp(&b.id))
        });

        output
    }


    fn select_neighbors(
        &self,
        center: usize,
        candidates: Vec<Candidate>,
        limit: usize,
    ) -> Vec<usize> {
        let _profile = ProfileTimer::new(
            &SELECT_NEIGHBORS_NS,
            &SELECT_NEIGHBORS_CALLS,
        );

        if limit == 0 {
            return Vec::new();
        }

        let mut selected = Vec::<usize>::with_capacity(limit);

        for candidate in &candidates {
            let slot = candidate.slot;

            if slot == center {
                continue;
            }

            let diverse = selected.iter().all(|&chosen| {
                let similarity = self.vectors[slot]
                    .normalized_dot(&self.vectors[chosen]);

                similarity < candidate.similarity
            });

            if diverse {
                selected.push(slot);

                if selected.len() == limit {
                    return selected;
                }
            }
        }

        // Fill remaining slots with the best candidates rejected by
        // the diversity heuristic.
        for candidate in &candidates {
            let slot = candidate.slot;

            if slot != center && !selected.contains(&slot) {
                selected.push(slot);

                if selected.len() == limit {
                    break;
                }
            }
        }

        selected
    }

    
    fn prune_neighbors(
        &mut self,
        center: usize,
        layer: usize,
        workspace: &mut PruneWorkspace,
    ) {
        if layer >= self.graph[center].len() {
            return;
        }

        let limit = self.m_for_layer(layer);
        if self.graph[center][layer].len() <= limit {
            return;
        }

        // Time only calls that actually prune an overflowing list.
        let _profile = ProfileTimer::new(
            &PRUNE_NEIGHBORS_NS,
            &PRUNE_NEIGHBORS_CALLS,
        );

        let candidate_count = self.graph[center][layer].len();
        workspace.prepare(candidate_count, limit);

        if workspace.selected_marks.len() < self.vectors.len() {
            workspace.selected_marks.resize(self.vectors.len(), 0);
        }

        workspace.generation = workspace.generation.wrapping_add(1);

        if workspace.generation == 0 {
            workspace.selected_marks.fill(0);
            workspace.generation = 1;
        }

        let generation = workspace.generation;

        let center_embedding = &self.vectors[center];

        for &slot in &self.graph[center][layer] {
            workspace.candidates.push(Candidate {
                slot,
                id: self.vectors[slot].id,
                similarity: center_embedding.normalized_dot(&self.vectors[slot]),
            });
        }

        if workspace.candidates.len() > 1 {
            workspace.candidates.sort_unstable_by(|a, b| {
                b.similarity
                    .total_cmp(&a.similarity)
                    .then_with(|| a.id.cmp(&b.id))
            });
        }

        // Diversity-first pass.
        for candidate in &workspace.candidates {
            let slot = candidate.slot;

            if slot == center {
                continue;
            }

            let diverse = workspace.selected.iter().all(|&chosen| {
                self.vectors[slot].normalized_dot(&self.vectors[chosen])
                    < candidate.similarity
            });

            if diverse {
                workspace.selected.push(slot);
                workspace.selected_marks[slot] = generation;

                if workspace.selected.len() == limit {
                    self.graph[center][layer].clone_from(&workspace.selected);
                    return;
                }
            }
        }

        // Fill remaining slots in similarity order.
        for candidate in &workspace.candidates {
            let slot = candidate.slot;

            if slot == center || workspace.selected_marks[slot] == generation {
                continue;
            }

            workspace.selected.push(slot);

            if workspace.selected.len() == limit {
                break;
            }
        }

        self.graph[center][layer].clone_from(&workspace.selected);
    }

}

fn random_level(id: VectorId, m: usize) -> usize {
    let mut x = id.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;

    let u = ((x >> 11) as f64 + 1.0)
        / ((1u64 << 53) as f64 + 1.0);

    ((-u.ln() / (m as f64).ln()).floor() as usize).min(MAX_LEVEL)
}

// -----------------------------------------------------------------------------
// Manual benchmark
// -----------------------------------------------------------------------------

#[cfg(test)]
mod benchmark {
    use super::*;
    use crate::index::VectorIndex;
    use std::collections::HashSet;
    use std::time::Instant;

    const DIM: usize = 128;
    const N: usize = 10_000;
    const QUERY_COUNT: usize = 100;
    const K: usize = 10;

    fn sample(mut x: u64) -> f32 {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;

        let unit = (x >> 40) as f32 / (1u32 << 24) as f32;
        unit * 2.0 - 1.0
    }

    fn vector(id: u64) -> Vec<f32> {
        (0..DIM)
            .map(|j| sample(id.wrapping_mul(DIM as u64).wrapping_add(j as u64)))
            .collect()
    }

    #[test]
    #[ignore = "manual benchmark; run with --ignored --nocapture"]
    fn benchmark_hnsw() {
        let mut exact = VectorIndex::new(DIM).unwrap();
        let mut hnsw = HnswIndex::new(DIM, 16, 200, 64).unwrap();

        let start = Instant::now();

        for i in 0..N {
            exact.insert(VectorId(i as u64), vector(i as u64)).unwrap();
        }

        let exact_build = start.elapsed();

        // Reset immediately before the HNSW build so only construction is measured.
        reset_profile();

        let start = Instant::now();

        for i in 0..N {
            hnsw.insert(VectorId(i as u64), vector(i as u64)).unwrap();
        }

        let hnsw_build = start.elapsed();

        println!("Vectors: {N}, dimensions: {DIM}, queries: {QUERY_COUNT}, k: {K}");
        println!("Exact build: {:?}", exact_build);
        println!("HNSW build:  {:?}", hnsw_build);

        // Print before search benchmarks so query-time calls do not affect these counters.
        print_profile_report();

        let queries: Vec<Vec<f32>> = (0..QUERY_COUNT)
            .map(|i| vector(50_000_000 + i as u64))
            .collect();

        // Compute exact ground truth once per query.
        let ground_truth: Vec<HashSet<VectorId>> = queries
            .iter()
            .map(|q| {
                exact
                    .search(q, K)
                    .unwrap()
                    .into_iter()
                    .map(|r| r.id)
                    .collect()
            })
            .collect();

        println!();
        println!(
            "{:>8} {:>12} {:>12} {:>12}",
            "ef", "recall@10", "avg us", "p95 us"
        );

        for ef in [16, 32, 64, 128, 200, 256] {
            let mut hits = 0usize;
            let mut latencies = Vec::with_capacity(QUERY_COUNT);

            for (i, query) in queries.iter().enumerate() {
                let start = Instant::now();
                let results = hnsw.search_with_ef(query, K, ef).unwrap();
                latencies.push(start.elapsed().as_secs_f64() * 1_000_000.0);

                hits += results
                    .iter()
                    .filter(|r| ground_truth[i].contains(&r.id))
                    .count();
            }

            latencies.sort_by(f64::total_cmp);

            let recall = hits as f64 / (QUERY_COUNT * K) as f64 * 100.0;
            let avg_us = latencies.iter().sum::<f64>() / latencies.len() as f64;
            let p95_us = latencies[((latencies.len() * 95).div_ceil(100) - 1)
                .min(latencies.len() - 1)];

            println!(
                "{ef:>8} {recall:>11.2}% {avg_us:>11.2} {p95_us:>11.2}"
            );
        }
    }
}