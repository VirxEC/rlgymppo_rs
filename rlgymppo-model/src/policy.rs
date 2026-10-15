use std::path::Path;

use burn::module::Module;
use burn::nn::modules::norm::NormalizationConfig;
use burn::nn::{LayerNormConfig, RmsNormConfig};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder, RecorderError};
use thiserror::Error;

use crate::Net;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormSelection {
    None,
    LayerNorm,
    RmsNorm,
}

#[derive(Clone, Debug)]
pub struct PolicyConfig {
    pub input_size: usize,
    pub action_size: usize,
    pub actor_layer_sizes: Vec<usize>,
    pub shared_head_layer_sizes: Vec<usize>,
    pub norm: NormSelection,
}

#[derive(Module, Debug)]
pub struct Policy<B: Backend> {
    pub shared_head: Option<Net<B>>,
    pub actor: Net<B>,
}

#[derive(Debug, Error)]
pub enum LoadPolicyError {
    #[error("policy hidden-layer lists must not be empty")]
    EmptyLayers,
    #[error("failed to load Burn policy record: {0}")]
    Recorder(#[from] RecorderError),
}

impl<B: Backend> Policy<B> {
    pub fn load(
        checkpoint: impl AsRef<Path>,
        config: &PolicyConfig,
        device: &B::Device,
    ) -> Result<Self, LoadPolicyError> {
        if config.actor_layer_sizes.is_empty() {
            return Err(LoadPolicyError::EmptyLayers);
        }

        let norm = normalization(config.norm);
        let actor_input = config
            .shared_head_layer_sizes
            .last()
            .copied()
            .unwrap_or(config.input_size);
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
        let checkpoint = checkpoint.as_ref();

        let actor = Net::<B>::new(
            actor_input,
            config.action_size,
            config.actor_layer_sizes.clone(),
            device,
            norm.clone(),
            true,
        )
        .load_file(checkpoint.join("actor"), &recorder, device)?;

        let shared_head = if config.shared_head_layer_sizes.is_empty() {
            None
        } else {
            Some(
                Net::<B>::new(
                    config.input_size,
                    0,
                    config.shared_head_layer_sizes.clone(),
                    device,
                    norm,
                    false,
                )
                .load_file(checkpoint.join("shared_head"), &recorder, device)?,
            )
        };

        Ok(Self { shared_head, actor })
    }

    /// Masked action logits (`actor.forward` over shared-head features plus
    /// the invalid-action bias). Exposed so distillation can run the expensive
    /// forward once and derive both the tempered and `T=1` distributions via
    /// [`Net::probs_with_temperature`].
    pub fn masked_logits(&self, input: Tensor<B, 2>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 2> {
        let features = match &self.shared_head {
            Some(head) => head.forward(input),
            None => input,
        };
        self.actor.masked_logits(features, mask)
    }

    /// Compute the stochastic action distribution `softmax(masked_logits)`
    /// clamped to `[1e-11, 1]`, optionally masking invalid actions with a large
    /// negative logit bias. Used as the teacher target distribution during
    /// transfer learning (knowledge distillation).
    pub fn infer(&self, input: Tensor<B, 2>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 2> {
        self.infer_with_temperature(input, mask, 1.0)
    }

    /// Temperature-scaled teacher distribution for distillation. `1.0`
    /// matches [`Policy::infer`]; higher values soften the targets.
    pub fn infer_with_temperature(
        &self,
        input: Tensor<B, 2>,
        mask: Option<Tensor<B, 2>>,
        temperature: f32,
    ) -> Tensor<B, 2> {
        let features = match &self.shared_head {
            Some(head) => head.forward(input),
            None => input,
        };
        self.actor
            .infer_with_temperature(features, mask, temperature)
    }

    #[must_use]
    pub fn react_deterministic_indexed(
        &self,
        observations: &[Vec<f32>],
        masks: &[Vec<bool>],
        indices: &[usize],
        device: &B::Device,
    ) -> Vec<usize> {
        let input = crate::tensor::to_state_tensor_2d_indexed(observations, indices, device);
        let features = match &self.shared_head {
            Some(head) => head.forward(input),
            None => input,
        };
        let mask = (!masks.is_empty())
            .then(|| crate::tensor::to_mask_tensor_2d_indexed(masks, indices, device));
        crate::tensor::argmax_actions(self.actor.masked_logits(features, mask))
    }
}

fn normalization(selection: NormSelection) -> Option<NormalizationConfig> {
    match selection {
        NormSelection::None => None,
        NormSelection::LayerNorm => Some(NormalizationConfig::Layer(LayerNormConfig::new(0))),
        NormSelection::RmsNorm => Some(NormalizationConfig::Rms(RmsNormConfig::new(0))),
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Flex;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};

    use super::*;
    use crate::Actic;

    #[test]
    fn loads_training_component_records() {
        let device = Default::default();
        let config = PolicyConfig {
            input_size: 5,
            action_size: 3,
            actor_layer_sizes: vec![4],
            shared_head_layer_sizes: vec![4],
            norm: NormSelection::RmsNorm,
        };
        let norm = normalization(config.norm);
        let model = Actic::<Flex>::new(
            config.input_size,
            config.action_size,
            config.actor_layer_sizes.clone(),
            vec![4],
            &config.shared_head_layer_sizes,
            &device,
            norm,
        );
        let directory = tempfile::tempdir().unwrap();
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
        model
            .actor
            .clone()
            .save_file(directory.path().join("actor"), &recorder)
            .unwrap();
        model
            .shared_head
            .as_ref()
            .unwrap()
            .clone()
            .save_file(directory.path().join("shared_head"), &recorder)
            .unwrap();

        let policy = Policy::<Flex>::load(directory.path(), &config, &device).unwrap();
        let observations = vec![vec![0.1, 0.2, 0.3, 0.4, 0.5]];
        let input = crate::tensor::to_state_tensor_2d(&observations, &device);
        let actual_features = policy.shared_head.as_ref().unwrap().forward(input);
        let actual = policy.actor.forward(actual_features).into_data();

        assert_eq!(
            actual.shape,
            burn::tensor::Shape::new([1, config.action_size])
        );
        assert!(
            actual
                .to_vec::<f32>()
                .unwrap()
                .into_iter()
                .all(f32::is_finite)
        );
    }

    #[test]
    fn masked_logits_matches_infer_paths() {
        let device = Default::default();
        let policy = Policy::<Flex> {
            shared_head: Some(Net::<Flex>::new(4, 0, vec![8], &device, None, false)),
            actor: Net::<Flex>::new(8, 3, vec![8], &device, None, true),
        };
        let input = Tensor::<Flex, 2>::ones([4, 4], &device);
        let logits = policy.masked_logits(input.clone(), None);
        let via_logits_t1 = Net::<Flex>::probs_with_temperature(logits.clone(), 1.0);
        let via_logits_hot = Net::<Flex>::probs_with_temperature(logits, 2.0);
        let direct_t1: Vec<f32> = policy
            .infer(input.clone(), None)
            .into_data()
            .into_vec()
            .unwrap();
        let direct_hot: Vec<f32> = policy
            .infer_with_temperature(input, None, 2.0)
            .into_data()
            .into_vec()
            .unwrap();
        let from_logits_t1: Vec<f32> = via_logits_t1.into_data().into_vec().unwrap();
        let from_logits_hot: Vec<f32> = via_logits_hot.into_data().into_vec().unwrap();
        assert_eq!(direct_t1, from_logits_t1);
        assert_eq!(direct_hot, from_logits_hot);
    }
}
