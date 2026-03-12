use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;
use ort::value::ValueType;

use crate::error::CanaryError;
use crate::vocab::Vocab;

/// Run autoregressive decoding using the decoder ONNX model.
///
/// This implements the standard encoder-decoder loop:
/// 1. Initialize an empty KV-cache (`decoder_mems`).
/// 2. On the first step, feed the full prompt as `input_ids`.
/// 3. On subsequent steps, feed only the last predicted token.
/// 4. Stop when the EOS token is emitted or `max_sequence_length` is reached.
///
/// Returns the decoded text string (special tokens filtered out).
pub fn decode_autoregressive(
    decoder: &mut Session,
    encoder_embeddings: &Tensor<f32>,
    encoder_mask: &Tensor<i64>,
    prompt_tokens: Vec<i64>,
    vocab: &Vocab,
    max_sequence_length: usize,
) -> Result<String, CanaryError> {
    // --- Determine decoder_mems shape from session metadata ---
    let (num_layers, hidden_dim) = extract_decoder_mems_shape(decoder)?;

    log::debug!(
        "Decoder cache dimensions: num_layers={}, hidden_dim={}",
        num_layers,
        hidden_dim
    );

    // Initial decoder_mems: [num_layers, batch=1, cache_len=0, hidden_dim]
    // ndarray supports zero-size dimensions natively, unlike Tensor::from_array with raw data.
    let empty_cache = Array4::<f32>::zeros((num_layers, 1, 0, hidden_dim));
    let mut decoder_mems = Tensor::from_array(empty_cache)?;

    let eos_id = vocab.eos_token_id();
    let mut all_tokens = prompt_tokens;
    let mut cache_len: usize = 0;

    log::debug!(
        "Starting autoregressive decode with {} prompt tokens, max_len={}",
        all_tokens.len(),
        max_sequence_length
    );

    for step in 0..max_sequence_length {
        // Build input_ids: full sequence on first call, last token only after that.
        let input_ids_tensor = if cache_len == 0 {
            // First step -- send entire prompt
            let len = all_tokens.len();
            let shape = vec![1i64, len as i64];
            Tensor::from_array((shape, all_tokens.clone().into_boxed_slice()))?
        } else {
            // Subsequent steps -- send only the most recent token
            let last = *all_tokens
                .last()
                .ok_or_else(|| CanaryError::InferenceError("Token list is empty".to_string()))?;
            Tensor::from_array((vec![1i64, 1i64], vec![last].into_boxed_slice()))?
        };

        // Run the decoder session
        let outputs = decoder.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "encoder_embeddings" => encoder_embeddings,
            "encoder_mask" => encoder_mask,
            "decoder_mems" => decoder_mems
        ])?;

        // Extract logits: [batch, seq_len, vocab_size]
        let (logits_shape, logits_data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| CanaryError::InferenceError(format!("Failed to extract logits: {e}")))?;

        // Greedy decode: argmax over the last time-step logits
        let logits_dims: Vec<i64> = logits_shape.iter().copied().collect();
        let seq_len = logits_dims[1] as usize;
        let vocab_size = logits_dims[2] as usize;

        let last_step_offset = (seq_len - 1) * vocab_size;
        let last_logits = &logits_data[last_step_offset..last_step_offset + vocab_size];

        let next_token = argmax(last_logits) as i64;

        log::debug!("Step {}: predicted token ID {}", step, next_token);

        if next_token == eos_id {
            log::debug!("EOS token reached at step {}", step);
            break;
        }

        all_tokens.push(next_token);

        // Extract updated decoder hidden states for KV-cache
        let (mems_shape_out, mems_data) = outputs["decoder_hidden_states"]
            .try_extract_tensor::<f32>()
            .map_err(|e| {
                CanaryError::InferenceError(format!("Failed to extract decoder_hidden_states: {e}"))
            })?;

        let new_shape: Vec<i64> = mems_shape_out.iter().copied().collect();
        cache_len = new_shape[2] as usize;

        decoder_mems = Tensor::from_array((new_shape, mems_data.to_vec().into_boxed_slice()))?;
    }

    let text = vocab.decode_tokens(&all_tokens);
    Ok(text)
}

/// Extract `num_layers` and `hidden_dim` from the decoder session's
/// `decoder_mems` input metadata.
///
/// The expected shape is `[num_layers, batch, cache_len, hidden_dim]` where
/// `num_layers` (dim 0) and `hidden_dim` (dim 3) are fixed dimensions, while
/// `batch` and `cache_len` may be dynamic (-1).
fn extract_decoder_mems_shape(decoder: &Session) -> Result<(usize, usize), CanaryError> {
    let mems_input = decoder
        .inputs
        .iter()
        .find(|outlet: &&ort::session::Input| outlet.name == "decoder_mems")
        .ok_or_else(|| {
            CanaryError::InferenceError("Decoder model missing 'decoder_mems' input".to_string())
        })?;

    match &mems_input.input_type {
        ValueType::Tensor { shape, .. } => {
            let dims: &[i64] = shape; // Shape derefs to [i64]
            if dims.len() != 4 {
                return Err(CanaryError::InferenceError(format!(
                    "Expected 4D decoder_mems, got {}D",
                    dims.len()
                )));
            }

            let num_layers = dims[0];
            let hidden_dim = dims[3];

            if num_layers <= 0 || hidden_dim <= 0 {
                return Err(CanaryError::InferenceError(format!(
                    "decoder_mems has dynamic num_layers ({}) or hidden_dim ({}); expected fixed",
                    num_layers, hidden_dim
                )));
            }

            Ok((num_layers as usize, hidden_dim as usize))
        }
        other => Err(CanaryError::InferenceError(format!(
            "decoder_mems input is not a tensor: {:?}",
            other
        ))),
    }
}

/// Return the index of the maximum element in a slice.
fn argmax(slice: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[-1.0, -3.0, -0.5]), 2);
        assert_eq!(argmax(&[5.0]), 0);
    }
}
