use crate::util::{compute::NonBlockingTransfer, sealed::Sealed};
use tch::{
    kind::FLOAT_CPU,
    nn::{self, LinearConfig, ModuleT},
    Device, Tensor,
};

pub struct ActionResult {
    pub action: Tensor,
    pub log_prob: Tensor,
}

pub struct BackpropResult {
    pub action_log_probs: Tensor,
    pub entropy: Tensor,
}

pub struct DiscretePolicy {
    seq: Sealed<nn::SequentialT>,
    device: Device,
}

impl DiscretePolicy {
    /// Min probability that an action will be taken
    pub const ACTION_MIN_PROB: f64 = 1e-11;

    pub fn new(
        input_size: i64,
        output_size: i64,
        layer_sizes: &[i64],
        path: nn::Path,
        device: Device,
    ) -> Self {
        let config = LinearConfig::default();

        let mut seq = nn::seq_t()
            .add(nn::linear(&path / 0, input_size, layer_sizes[0], config))
            .add_fn(|xs| xs.relu());

        let mut prev_layer_size = layer_sizes[0];
        for (i, layer_size) in layer_sizes.iter().skip(1).copied().enumerate() {
            seq = seq
                .add(nn::linear(
                    &path / (i + 1),
                    prev_layer_size,
                    layer_size,
                    config,
                ))
                .add_fn(|xs| xs.relu());
            prev_layer_size = layer_size;
        }

        // output layer, one neuron for each action
        seq = seq
            .add(nn::linear(
                path / layer_sizes.len(),
                prev_layer_size,
                output_size,
                config,
            ))
            .add_fn(|xs| xs.softmax(-1, tch::Kind::Float));

        Self {
            seq: Sealed::new(seq),
            device,
        }
    }

    fn get_output(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
        self.seq.forward_t(xs, train)
    }

    fn get_action_probs(&self, obs: &Tensor, train: bool) -> Tensor {
        let mut probs = self.get_output(obs, train);
        probs = probs.view((-1i64, probs.size()[1]));
        probs = probs.clamp(Self::ACTION_MIN_PROB, 1.0);
        probs
    }

    pub fn get_action(&self, obs: &Tensor, deterministic: bool) -> ActionResult {
        let probs = self.get_action_probs(obs, deterministic);

        if deterministic {
            let action = probs.argmax(1, true);
            let log_prob = Tensor::zeros(action.numel() as i64, FLOAT_CPU);

            ActionResult {
                action: action.to(Device::Cpu).flatten(0, -1),
                log_prob,
            }
        } else {
            let action = probs.multinomial(1, true);
            let log_prob = probs.log().gather(-1, &action, false);

            ActionResult {
                action: action.to(Device::Cpu).flatten(0, -1),
                log_prob: log_prob.to(Device::Cpu).flatten(0, -1),
            }
        }
    }

    pub fn get_backprop_data(&self, obs: &Tensor, acts: &Tensor) -> BackpropResult {
        let acts = acts.to_dtype(tch::Kind::Int64, true, false);
        let probs = self.get_action_probs(obs, true);

        let log_probs = probs.log();
        let action_log_probs = log_probs.gather(-1, &acts, false);
        let entropy = -(log_probs * probs).sum(None);

        BackpropResult {
            action_log_probs: action_log_probs.no_block_to(self.device),
            entropy: entropy.no_block_to(self.device).mean(None),
        }
    }
}
