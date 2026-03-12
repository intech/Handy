use thiserror::Error;

/// Errors that can occur during Canary model operations.
#[derive(Debug, Error)]
pub enum CanaryError {
    /// One or more ONNX model sessions have not been loaded.
    #[error("Model not loaded: call load_model() first")]
    ModelNotLoaded,

    /// An error originating from the ONNX Runtime.
    #[error("ONNX runtime error: {0}")]
    OnnxError(String),

    /// An error related to vocabulary loading or token lookup.
    #[error("Vocabulary error: {0}")]
    VocabError(String),

    /// An error during the inference pipeline.
    #[error("Inference error: {0}")]
    InferenceError(String),
}

impl From<ort::Error> for CanaryError {
    fn from(e: ort::Error) -> Self {
        CanaryError::OnnxError(e.to_string())
    }
}
