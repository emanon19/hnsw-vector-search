use crate::error::SearchError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VectorId(pub u64);

#[derive(Debug, Clone)]
pub struct Embedding {
    pub id: VectorId,
    pub data: Vec<f32>,
    pub norm: f64,
}

impl Embedding {
    pub fn new(id: VectorId, data: Vec<f32>) -> Result<Self, SearchError> {
        if data.is_empty() {
            return Err(SearchError::EmptyVector);
        }

        if data.iter().any(|x| !x.is_finite()) {
            return Err(SearchError::NonFiniteValue);
        }

        let norm_squared: f64 = data
            .iter()
            .map(|&x| {
                let x = f64::from(x);
                x * x
            })
            .sum();

        let norm = norm_squared.sqrt();

        if norm == 0.0 || !norm.is_finite() {
            return Err(SearchError::ZeroNorm);
        }

        Ok(Self { id, data, norm })
    }

    #[inline]
    pub fn cosine_similarity(&self, query: &[f32], query_norm: f64) -> f32 {
        let dot_product: f64 = self
            .data
            .iter()
            .zip(query.iter())
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();

        (dot_product / (self.norm * query_norm)) as f32
    }
}
