use crate::error::SearchError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VectorId(pub u64);

#[derive(Debug, Clone)]
pub struct Embedding {
    pub id: VectorId,
    pub data: Vec<f32>,
    pub norm: f64,
    pub inv_norm: f64,
    pub normalized: Vec<f32>,
}

impl Embedding {
    #[inline]
    pub fn new(id: VectorId, data: Vec<f32>) -> Result<Self, SearchError> {
        if data.is_empty() {
            return Err(SearchError::EmptyVector);
        }

        let mut norm_squared = 0.0_f64;

        for &x in &data {
            if !x.is_finite() {
                return Err(SearchError::NonFiniteValue);
            }

            let value = f64::from(x);
            norm_squared += value * value;
        }

        let norm = norm_squared.sqrt();

        if norm == 0.0 || !norm.is_finite() {
            return Err(SearchError::ZeroNorm);
        }

        let normalized = data
            .iter()
            .map(|&x| (f64::from(x) / norm) as f32)
            .collect();

        let inv_norm = 1.0 / norm;

        Ok(Self { 
            id, 
            data, 
            norm,
            inv_norm,
            normalized
        })
    }

    #[inline]
    pub fn cosine_similarity(&self, query: &[f32], query_norm: f64) -> f32 {
        debug_assert_eq!(self.data.len(), query.len());

        let mut dot = 0.0_f64;
        let mut i = 0;

        while i + 4 <= self.data.len() {
            dot += f64::from(self.data[i]) * f64::from(query[i]);
            dot += f64::from(self.data[i + 1]) * f64::from(query[i + 1]);
            dot += f64::from(self.data[i + 2]) * f64::from(query[i + 2]);
            dot += f64::from(self.data[i + 3]) * f64::from(query[i + 3]);
            i += 4;
        }

        while i < self.data.len() {
            dot += f64::from(self.data[i]) * f64::from(query[i]);
            i += 1;
        }

        (dot * self.inv_norm / query_norm) as f32
    }

    #[inline]
    pub fn normalized_dot(&self, other: &Self) -> f32 {
        let a = &self.normalized;
        let b = &other.normalized;

        let len = a.len();
        let mut i = 0;

        let mut sum0 = 0.0_f32;
        let mut sum1 = 0.0_f32;
        let mut sum2 = 0.0_f32;
        let mut sum3 = 0.0_f32;

        while i + 4 <= len {
            sum0 += a[i] * b[i];
            sum1 += a[i + 1] * b[i + 1];
            sum2 += a[i + 2] * b[i + 2];
            sum3 += a[i + 3] * b[i + 3];
            i += 4;
        }

        let mut sum = (sum0 + sum1) + (sum2 + sum3);

        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }

        sum
    }
}