use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::error::SearchError;
use crate::index::SearchResult;
use crate::vector::{Embedding, VectorId};

const MAX_LEVEL: usize = 16;

#[derive(Debug, Clone, Copy)]
struct Candidate {
    slot: usize,
    id: VectorId,
    score: f32,
}

// Max-heap: highest similarity first; lower ID wins ties.
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score) == Ordering::Equal
            && self.id == other.id
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
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.id.0.cmp(&self.id.0))
    }
}

impl Candidate {
    fn better_than(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score) == Ordering::Greater
            || (self.score.total_cmp(&other.score) == Ordering::Equal
                && self.id < other.id)
    }
}

pub struct HnswIndex {
    dimension: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    vectors: Vec<Embedding>,
    id_to_slot: HashMap<VectorId, usize>,
    levels: Vec<usize>,
    // graph[node][layer] stores neighboring vector slots.
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
        if dimension == 0 {
            return Err(SearchError::EmptyVector);
        }

        if m < 2 || ef_construction < m || ef_search == 0 {
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

        // Validate the embedding before mutating the index.
        let embedding = Embedding::new(id, data)?;
        let level = Self::random_level(id, self.m);
        let slot = self.vectors.len();

        self.vectors.push(embedding);
        self.levels.push(level);
        self.graph.push(vec![Vec::new(); level + 1]);
        self.id_to_slot.insert(id, slot);

        let Some(mut current) = self.entry_point else {
            self.entry_point = Some(slot);
            self.max_level = level;
            return Ok(());
        };

        let query = self.vectors[slot].data.clone();
        let query_norm = self.vectors[slot].norm;
        let old_max_level = self.max_level;

        // Greedily descend through layers the new node cannot join.
        if level < old_max_level {
            for layer in ((level + 1)..=old_max_level).rev() {
                current = self.greedy_search(&query, query_norm, current, layer);
            }
        }

        // Search from upper to lower layers.
        for layer in (0..=level.min(old_max_level)).rev() {
            let candidates = self.search_layer(
                &query,
                query_norm,
                &[current],
                self.ef_construction,
                layer,
            );

            let limit = self.max_neighbors(layer);
            let neighbors =
                self.select_neighbors(slot, candidates, limit);

            // Establish reciprocal links before pruning the existing nodes.
            self.graph[slot][layer] = neighbors.clone();

            for &neighbor in &neighbors {
                self.add_directed_edge(neighbor, slot, layer);
            }

            for neighbor in neighbors {
                if self.graph[neighbor][layer].len() > self.max_neighbors(layer) {
                    self.prune_neighbors(neighbor, layer);
                }
            }

            if let Some(&nearest) = self.graph[slot][layer].first() {
                current = nearest;
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
        self.validate_query(query)?;

        if top_k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }

        let query_norm = norm(query);
        let mut current = self.entry_point.expect("non-empty index has entry");

        // Greedy descent through the upper layers.
        for layer in (1..=self.max_level).rev() {
            current = self.greedy_search(query, query_norm, current, layer);
        }

        let candidates = self.search_layer(
            query,
            query_norm,
            &[current],
            ef.max(top_k).max(1),
            0,
        );

        Ok(candidates
            .into_iter()
            .take(top_k)
            .map(|candidate| SearchResult {
                id: candidate.id,
                similarity: candidate.score,
            })
            .collect())
    }

    fn validate_query(&self, query: &[f32]) -> Result<(), SearchError> {
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        if query.iter().any(|value| !value.is_finite()) {
            return Err(SearchError::NonFiniteValue);
        }

        if norm(query) == 0.0 {
            return Err(SearchError::ZeroNorm);
        }

        Ok(())
    }

    fn score(&self, slot: usize, query: &[f32], query_norm: f64) -> f32 {
        self.vectors[slot].cosine_similarity(query, query_norm)
    }

    fn max_neighbors(&self, layer: usize) -> usize {
        if layer == 0 {
            self.m.saturating_mul(2)
        } else {
            self.m
        }
    }

    fn greedy_search(
        &self,
        query: &[f32],
        query_norm: f64,
        start: usize,
        layer: usize,
    ) -> usize {
        let mut current = start;
        let mut current_score = Candidate {
            slot: current,
            id: self.vectors[current].id,
            score: self.score(current, query, query_norm),
        };

        loop {
            let mut best = current_score;

            for &neighbor in &self.graph[current][layer] {
                let candidate = Candidate {
                    slot: neighbor,
                    id: self.vectors[neighbor].id,
                    score: self.score(neighbor, query, query_norm),
                };

                if candidate.better_than(&best) {
                    best = candidate;
                }
            }

            if best.slot == current {
                return current;
            }

            current = best.slot;
            current_score = best;
        }
    }

    // Best-first graph traversal with a bounded result set.
    fn search_layer(
        &self,
        query: &[f32],
        query_norm: f64,
        entries: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        if ef == 0 {
            return Vec::new();
        }

        // Best candidate to explore is popped first.
        let mut exploration = BinaryHeap::<Candidate>::new();

        // Reverse makes the worst retained candidate the heap maximum.
        let mut results = BinaryHeap::<Reverse<Candidate>>::new();
        let mut visited = HashSet::new();

        for &slot in entries {
            if !visited.insert(slot) {
                continue;
            }

            let candidate = Candidate {
                slot,
                id: self.vectors[slot].id,
                score: self.score(slot, query, query_norm),
            };

            exploration.push(candidate);
            results.push(Reverse(candidate));
        }

        while let Some(candidate) = exploration.pop() {
            if results.len() >= ef {
                let worst = results
                    .peek()
                    .expect("results is non-empty")
                    .0;

                // All remaining exploration candidates are no better.
                if !candidate.better_than(&worst) {
                    break;
                }
            }

            for &neighbor in &self.graph[candidate.slot][layer] {
                if !visited.insert(neighbor) {
                    continue;
                }

                let next = Candidate {
                    slot: neighbor,
                    id: self.vectors[neighbor].id,
                    score: self.score(neighbor, query, query_norm),
                };

                let should_add = results.len() < ef
                    || next.better_than(
                        &results.peek().expect("results is non-empty").0,
                    );

                if should_add {
                    exploration.push(next);
                    results.push(Reverse(next));

                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut output: Vec<Candidate> = results
            .into_iter()
            .map(|Reverse(candidate)| candidate)
            .collect();

        output.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.id.cmp(&b.id))
        });

        output
    }

    // Select neighbors using the HNSW diversity heuristic.
    fn select_neighbors(
        &self,
        center: usize,
        mut candidates: Vec<Candidate>,
        limit: usize,
    ) -> Vec<usize> {
        candidates.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.id.cmp(&b.id))
        });

        // Remove duplicate slots while preserving candidate ranking.
        let mut seen = HashSet::with_capacity(candidates.len());
        candidates.retain(|candidate| seen.insert(candidate.slot));

        let center_data = &self.vectors[center].data;
        let center_norm = self.vectors[center].norm;

        let mut selected: Vec<Candidate> = Vec::with_capacity(limit);
        let mut deferred: Vec<Candidate> = Vec::new();

        for candidate in candidates {
            if candidate.slot == center {
                continue;
            }

            if selected.len() >= limit {
                break;
            }

            let center_similarity =
                self.score(candidate.slot, center_data, center_norm);

            let diverse = selected.iter().all(|chosen| {
                let between_similarity = self.vectors[candidate.slot]
                    .cosine_similarity(
                        &self.vectors[chosen.slot].data,
                        self.vectors[chosen.slot].norm,
                    );

                // Cosine-distance form:
                // d(candidate, chosen) > d(candidate, center)
                // is equivalent to between_similarity < center_similarity.
                between_similarity < center_similarity
            });

            if diverse {
                selected.push(candidate);
            } else {
                deferred.push(candidate);
            }
        }

        // Fill remaining capacity with the closest deferred candidates.
        for candidate in deferred {
            if selected.len() >= limit {
                break;
            }

            selected.push(candidate);
        }

        selected
            .into_iter()
            .map(|candidate| candidate.slot)
            .collect()
    }

    fn add_directed_edge(&mut self, from: usize, to: usize, layer: usize) {
        if from == to || self.graph[from][layer].contains(&to) {
            return;
        }

        self.graph[from][layer].push(to);
    }

    fn prune_neighbors(&mut self, center: usize, layer: usize) {
        let candidates: Vec<Candidate> = self.graph[center][layer]
            .iter()
            .copied()
            .map(|slot| Candidate {
                slot,
                id: self.vectors[slot].id,
                score: self.vectors[center].cosine_similarity(
                    &self.vectors[slot].data,
                    self.vectors[slot].norm,
                ),
            })
            .collect();

        let selected =
            self.select_neighbors(center, candidates, self.max_neighbors(layer));

        self.graph[center][layer] = selected;
    }

    // Deterministic logarithmic level distribution:
    // P(level >= k) is approximately 1 / M^k.
    fn random_level(id: VectorId, m: usize) -> usize {
        let mut x = id.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;

        let mantissa = x >> 11;
        let u = (mantissa as f64 + 1.0) / ((1u64 << 53) as f64);

        ((-u.ln() / (m as f64).ln()).floor() as usize).min(MAX_LEVEL)
    }
}

fn norm(vector: &[f32]) -> f64 {
    vector
        .iter()
        .map(|&value| {
            let value = f64::from(value);
            value * value
        })
        .sum::<f64>()
        .sqrt()
}