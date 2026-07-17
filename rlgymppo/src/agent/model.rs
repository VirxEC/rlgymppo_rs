use burn::module::ParamId;
use burn::tensor::backend::Backend;
pub use rlgymppo_model::{Actic, Net, PPOOutput};

/// Return the parameter IDs of the first `hidden_layers` linear weights in a
/// `Net`. The vendored model stores hidden layers before its optional output
/// layer, so output weights are excluded when `hidden_layers` is set to the
/// number of hidden layers.
pub fn linear_weight_param_ids<B: Backend>(net: &Net<B>, hidden_layers: usize) -> Vec<ParamId> {
    net.linear_layers()
        .iter()
        .take(hidden_layers)
        .map(|layer| layer.weight.id)
        .collect()
}
