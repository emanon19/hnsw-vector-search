use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchError {
    EmptyVector,
    NonFiniteValue,
    ZeroNorm,
    DimensionMismatch { expected: usize, actual: usize },
    DuplicateId(u64),
    InvalidWorkerCount,
    WorkerPanicked,
    InvalidHnswConfig,
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyVector => write!(f, "vector must not be empty"),
            Self::NonFiniteValue => write!(f, "vector contains a non-finite value"),
            Self::ZeroNorm => write!(f, "vector must have a non-zero norm"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::DuplicateId(id) => write!(f, "duplicate vector ID: {id}"),
            Self::InvalidWorkerCount => write!(f, "worker count must be greater than zero"),
            Self::WorkerPanicked => write!(f, "a search worker panicked"),
            Self::InvalidHnswConfig => {
                write!(f, "invalid HNSW configuration")
            }
        }
    }
}

impl std::error::Error for SearchError {}
