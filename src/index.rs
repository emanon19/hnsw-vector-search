use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::thread;

use crate::error::SearchError;
use crate::vector::{Embedding, VectorId};

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: VectorId,
    pub similarity: f32,
}

pub struct VectorIndex {
    dimension: usize,
    vectors: Vec<Embedding>,
    id_to_slot: HashMap<VectorId, usize>,
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    id: VectorId,
    similarity: f32,
}

// BinaryHeap is a max-heap. Reverse score ordering so the worst
// candidate is at the root and can be replaced efficiently.
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.similarity.total_cmp(&other.similarity) == Ordering::Equal && self.id == other.id
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
        other
            .similarity
            .total_cmp(&self.similarity)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl Candidate {
    fn is_better_than(&self, other: &Self) -> bool {
        self.similarity.total_cmp(&other.similarity) == Ordering::Greater
            || (self.similarity.total_cmp(&other.similarity) == Ordering::Equal
                && self.id < other.id)
    }
}

impl VectorIndex {
    pub fn new(dimension: usize) -> Result<Self, SearchError> {
        if dimension == 0 {
            return Err(SearchError::EmptyVector);
        }

        Ok(Self {
            dimension,
            vectors: Vec::new(),
            id_to_slot: HashMap::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn insert(&mut self, id: VectorId, data: Vec<f32>) -> Result<(), SearchError> {
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
        let slot = self.vectors.len();

        self.vectors.push(embedding);
        self.id_to_slot.insert(id, slot);

        Ok(())
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchResult>, SearchError> {
        let query_norm = self.prepare_query(query, top_k)?;
        if top_k == 0 || self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let candidates = top_k_candidates(
            &self.vectors,
            query,
            query_norm,
            top_k.min(self.vectors.len()),
        );

        Ok(finish_candidates(candidates))
    }

    pub fn search_concurrent(
        &self,
        query: &[f32],
        top_k: usize,
        worker_count: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        if worker_count == 0 {
            return Err(SearchError::InvalidWorkerCount);
        }

        let query_norm = self.prepare_query(query, top_k)?;
        if top_k == 0 || self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let k = top_k.min(self.vectors.len());
        let chunk_size = self.vectors.len().div_ceil(worker_count);

        // Scoped threads borrow the index and query; no full-index clone or Arc.
        let partials: Result<Vec<Vec<Candidate>>, SearchError> = thread::scope(|scope| {
            let mut handles = Vec::new();

            for chunk in self.vectors.chunks(chunk_size) {
                handles.push(scope.spawn(move || top_k_candidates(chunk, query, query_norm, k)));
            }

            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| SearchError::WorkerPanicked))
                .collect()
        });

        let mut merged = BinaryHeap::with_capacity(k);
        for candidates in partials? {
            for candidate in candidates {
                push_bounded(&mut merged, candidate, k);
            }
        }

        Ok(finish_candidates(merged.into_vec()))
    }

    fn prepare_query(&self, query: &[f32], top_k: usize) -> Result<f64, SearchError> {
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        if top_k == 0 || self.vectors.is_empty() {
            return Ok(0.0);
        }

        let mut squared_norm = 0.0_f64;

        for &value in query {
            if !value.is_finite() {
                return Err(SearchError::NonFiniteValue);
            }

            let value = f64::from(value);
            squared_norm += value * value;
        }

        if squared_norm == 0.0 {
            return Err(SearchError::ZeroNorm);
        }

        Ok(squared_norm.sqrt())
    }
}

fn top_k_candidates(
    vectors: &[Embedding],
    query: &[f32],
    query_norm: f64,
    k: usize,
) -> Vec<Candidate> {
    let mut heap = BinaryHeap::with_capacity(k);

    for vector in vectors {
        let candidate = Candidate {
            id: vector.id,
            similarity: vector.cosine_similarity(query, query_norm),
        };

        push_bounded(&mut heap, candidate, k);
    }

    heap.into_vec()
}

fn push_bounded(heap: &mut BinaryHeap<Candidate>, candidate: Candidate, k: usize) {
    if heap.len() < k {
        heap.push(candidate);
        return;
    }

    if heap
        .peek()
        .is_some_and(|worst| candidate.is_better_than(worst))
    {
        heap.pop();
        heap.push(candidate);
    }
}

fn finish_candidates(mut candidates: Vec<Candidate>) -> Vec<SearchResult> {
    candidates.sort_unstable_by(|a, b| {
        b.similarity
            .total_cmp(&a.similarity)
            .then_with(|| a.id.cmp(&b.id))
    });

    candidates
        .into_iter()
        .map(|candidate| SearchResult {
            id: candidate.id,
            similarity: candidate.similarity,
        })
        .collect()
}
