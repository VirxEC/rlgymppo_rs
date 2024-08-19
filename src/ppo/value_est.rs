use crate::util::sealed::Sealed;
use tch::{
    nn::{self, ModuleT},
    Device, Tensor,
};

pub struct ValueEstimator {
    seq: Sealed<nn::SequentialT>,
    device: Device,
}

impl ValueEstimator {
    pub fn new(input_size: i64, layer_sizes: &[i64], path: nn::Path, device: Device) -> Self {
        let mut seq = nn::seq_t()
            .add(nn::linear(
                &path / 0,
                input_size,
                layer_sizes[0],
                Default::default(),
            ))
            .add_fn(|xs| xs.relu());

        let mut prev_layer_size = layer_sizes[0];
        for (i, layer_size) in layer_sizes.iter().skip(1).copied().enumerate() {
            seq = seq
                .add(nn::linear(
                    &path / (i + 1),
                    prev_layer_size,
                    layer_size,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu());
            prev_layer_size = layer_size;
        }

        // Output layer, just gives 1 output for value estimate
        seq = seq.add(nn::linear(
            path / layer_sizes.len(),
            prev_layer_size,
            1,
            Default::default(),
        ));

        Self {
            seq: Sealed::new(seq),
            device,
        }
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Tensor {
        self.seq.forward_t(xs, train).to(self.device)
    }
}
