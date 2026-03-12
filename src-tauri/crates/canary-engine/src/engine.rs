use std::path::Path;
use std::time::Instant;

use ort::session::Session;
use ort::value::Tensor;

use crate::decoder::decode_autoregressive;
use crate::error::CanaryError;
use crate::vocab::Vocab;

/// Configuration for a transcription request.
pub struct CanaryConfig {
    /// Whether to include punctuation in the output.
    pub use_pnc: bool,
    /// Maximum number of tokens the decoder will generate.
    pub max_sequence_length: usize,
}

impl Default for CanaryConfig {
    fn default() -> Self {
        Self {
            use_pnc: true,
            max_sequence_length: 1024,
        }
    }
}

/// The result of a transcription.
pub struct TranscriptionResult {
    /// Decoded text output.
    pub text: String,
}

/// Canary 1B v2 speech-to-text inference engine.
///
/// Manages three ONNX Runtime sessions (preprocessor, encoder, decoder) and
/// a vocabulary. Call [`load_model`](Self::load_model) before
/// [`transcribe`](Self::transcribe).
pub struct CanaryEngine {
    preprocessor: Option<Session>,
    encoder: Option<Session>,
    decoder: Option<Session>,
    vocab: Option<Vocab>,
}

impl CanaryEngine {
    /// Create a new engine with no models loaded.
    pub fn new() -> Self {
        Self {
            preprocessor: None,
            encoder: None,
            decoder: None,
            vocab: None,
        }
    }

    /// Load the three ONNX models and vocabulary from a directory.
    ///
    /// Expected files inside `model_path`:
    /// - `nemo128.onnx`            (preprocessor)
    /// - `encoder-model.int8.onnx` (encoder)
    /// - `decoder-model.int8.onnx` (decoder)
    /// - `vocab.txt`               (token vocabulary)
    pub fn load_model(&mut self, model_path: &Path) -> Result<(), CanaryError> {
        let start = Instant::now();

        log::info!("Loading Canary models from {}", model_path.display());

        let preprocessor_path = model_path.join("nemo128.onnx");
        let encoder_path = model_path.join("encoder-model.int8.onnx");
        let decoder_path = model_path.join("decoder-model.int8.onnx");
        let vocab_path = model_path.join("vocab.txt");

        // Validate all files exist before loading anything
        for (name, path) in [
            ("preprocessor", &preprocessor_path),
            ("encoder", &encoder_path),
            ("decoder", &decoder_path),
            ("vocab", &vocab_path),
        ] {
            if !path.exists() {
                return Err(CanaryError::OnnxError(format!(
                    "{name} file not found: {}",
                    path.display()
                )));
            }
        }

        log::info!("Loading preprocessor: {}", preprocessor_path.display());
        let preprocessor = Session::builder()?.commit_from_file(&preprocessor_path)?;

        log::info!("Loading encoder: {}", encoder_path.display());
        let encoder = Session::builder()?.commit_from_file(&encoder_path)?;

        log::info!("Loading decoder: {}", decoder_path.display());
        let decoder = Session::builder()?.commit_from_file(&decoder_path)?;

        let vocab = Vocab::load(&vocab_path)?;

        self.preprocessor = Some(preprocessor);
        self.encoder = Some(encoder);
        self.decoder = Some(decoder);
        self.vocab = Some(vocab);

        log::info!("Canary models loaded in {:.2?}", start.elapsed());

        Ok(())
    }

    /// Unload all models and free associated resources.
    pub fn unload_model(&mut self) {
        self.preprocessor = None;
        self.encoder = None;
        self.decoder = None;
        self.vocab = None;
        log::info!("Canary models unloaded");
    }

    /// Returns `true` if all three models and the vocabulary are loaded.
    pub fn is_loaded(&self) -> bool {
        self.preprocessor.is_some()
            && self.encoder.is_some()
            && self.decoder.is_some()
            && self.vocab.is_some()
    }

    /// Transcribe raw audio samples into text.
    ///
    /// # Arguments
    /// * `audio_samples` -- PCM f32 samples (mono, 16 kHz expected by the preprocessor model).
    /// * `source_language` -- BCP-47 language tag for the audio (defaults to `"en"`).
    /// * `target_language` -- BCP-47 language tag for the output text (defaults to `"en"`).
    /// * `config` -- Decoding configuration.
    pub fn transcribe(
        &mut self,
        audio_samples: Vec<f32>,
        source_language: Option<&str>,
        target_language: Option<&str>,
        config: &CanaryConfig,
    ) -> Result<TranscriptionResult, CanaryError> {
        let preprocessor = self
            .preprocessor
            .as_mut()
            .ok_or(CanaryError::ModelNotLoaded)?;
        let encoder = self.encoder.as_mut().ok_or(CanaryError::ModelNotLoaded)?;
        let decoder = self.decoder.as_mut().ok_or(CanaryError::ModelNotLoaded)?;
        let vocab = self.vocab.as_ref().ok_or(CanaryError::ModelNotLoaded)?;

        let src_lang = source_language.unwrap_or("en");
        let tgt_lang = target_language.unwrap_or("en");

        let total_start = Instant::now();

        // --- Step 1: Preprocess audio -> mel features ---
        let preprocess_start = Instant::now();
        let num_samples = audio_samples.len();

        log::debug!("Preprocessor input: waveforms shape [1, {}]", num_samples);

        let waveforms = Tensor::from_array((
            vec![1i64, num_samples as i64],
            audio_samples.into_boxed_slice(),
        ))?;
        let waveforms_lens =
            Tensor::from_array((vec![1i64], vec![num_samples as i64].into_boxed_slice()))?;

        let preprocess_out = preprocessor.run(ort::inputs![
            "waveforms" => waveforms,
            "waveforms_lens" => waveforms_lens
        ])?;

        // features: [batch, 128, time_frames]
        // features_lens: [batch]
        // We need to pass these onward as Tensor values for the encoder.
        let (features_shape, features_data) = preprocess_out["features"]
            .try_extract_tensor::<f32>()
            .map_err(|e| CanaryError::InferenceError(format!("Failed to extract features: {e}")))?;
        let (features_lens_shape, features_lens_data) = preprocess_out["features_lens"]
            .try_extract_tensor::<i64>()
            .map_err(|e| {
                CanaryError::InferenceError(format!("Failed to extract features_lens: {e}"))
            })?;

        let features_shape_vec: Vec<i64> = features_shape.iter().copied().collect();
        let features_lens_shape_vec: Vec<i64> = features_lens_shape.iter().copied().collect();

        log::debug!(
            "Preprocessor output: features shape {:?}, lens {:?} ({:.2?})",
            features_shape_vec,
            features_lens_data,
            preprocess_start.elapsed()
        );

        let features_tensor = Tensor::from_array((
            features_shape_vec,
            features_data.to_vec().into_boxed_slice(),
        ))?;
        let features_lens_tensor = Tensor::from_array((
            features_lens_shape_vec,
            features_lens_data.to_vec().into_boxed_slice(),
        ))?;

        // --- Step 2: Encode mel features -> encoder embeddings ---
        let encode_start = Instant::now();

        let encoder_out = encoder.run(ort::inputs![
            "audio_signal" => features_tensor,
            "length" => features_lens_tensor
        ])?;

        // encoder_embeddings: [batch, enc_time, hidden_dim]
        // encoder_mask: [batch, enc_time]
        let (enc_emb_shape, enc_emb_data) = encoder_out["encoder_embeddings"]
            .try_extract_tensor::<f32>()
            .map_err(|e| {
                CanaryError::InferenceError(format!("Failed to extract encoder_embeddings: {e}"))
            })?;
        let (enc_mask_shape, enc_mask_data) = encoder_out["encoder_mask"]
            .try_extract_tensor::<i64>()
            .map_err(|e| {
                CanaryError::InferenceError(format!("Failed to extract encoder_mask: {e}"))
            })?;

        let enc_emb_shape_vec: Vec<i64> = enc_emb_shape.iter().copied().collect();
        let enc_mask_shape_vec: Vec<i64> = enc_mask_shape.iter().copied().collect();

        log::debug!(
            "Encoder output: embeddings shape {:?}, mask shape {:?} ({:.2?})",
            enc_emb_shape_vec,
            enc_mask_shape_vec,
            encode_start.elapsed()
        );

        let encoder_embeddings =
            Tensor::from_array((enc_emb_shape_vec, enc_emb_data.to_vec().into_boxed_slice()))?;
        let encoder_mask = Tensor::from_array((
            enc_mask_shape_vec,
            enc_mask_data.to_vec().into_boxed_slice(),
        ))?;

        // --- Step 3: Build prompt tokens ---
        let prompt_tokens = vocab.build_prompt(src_lang, tgt_lang, config.use_pnc)?;

        log::debug!(
            "Prompt tokens ({}): {:?}",
            prompt_tokens.len(),
            prompt_tokens
        );

        // --- Step 4: Autoregressive decoding ---
        let decode_start = Instant::now();

        let text = decode_autoregressive(
            decoder,
            &encoder_embeddings,
            &encoder_mask,
            prompt_tokens,
            vocab,
            config.max_sequence_length,
        )?;

        log::debug!("Decoding completed in {:.2?}", decode_start.elapsed());
        log::info!(
            "Transcription completed in {:.2?}: \"{}\"",
            total_start.elapsed(),
            text
        );

        Ok(TranscriptionResult { text })
    }
}

impl Default for CanaryEngine {
    fn default() -> Self {
        Self::new()
    }
}
