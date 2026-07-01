pub mod actions;
pub mod obs;
pub mod rewards;
pub mod shared_info;
pub mod state_setters;
pub mod terminal;

mod avg_tracker;
mod report;
pub(crate) mod running_stat;
pub(crate) mod serde;

pub use avg_tracker::AvgTracker;
use burn::prelude::*;
use burn::tensor::cast::ToElement;
use burn::tensor::{Distribution, Transaction};
pub use report::{Report, Reportable};

pub(crate) fn to_mask_tensor_2d<B: Backend>(
    masks: &[Vec<bool>],
    device: &B::Device,
) -> Tensor<B, 2> {
    let shape = [masks.len(), masks[0].len()];
    let data: Vec<f32> = masks
        .iter()
        .flat_map(|m| m.iter().map(|&v| if v { 1.0 } else { 0.0 }))
        .collect();
    Tensor::from_data(TensorData::new(data, shape), device)
}

pub(crate) fn to_state_tensor_2d<B: Backend>(
    state: &[Vec<f32>],
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut data: Vec<f32> = Vec::with_capacity(state.len() * state[0].len());
    for elem in state {
        data.extend(elem);
    }

    Tensor::from_data(TensorData::new(data, [state.len(), state[0].len()]), device)
}

pub(crate) fn sample_actions<B: Backend>(
    action_probs: Tensor<B, 2>,
    device: &B::Device,
) -> (Vec<usize>, Vec<f32>) {
    let shape = action_probs.shape();
    let log_probs = action_probs.log();

    let u = Tensor::<B, 2>::random(shape, Distribution::Default, device);
    let gumbel = u.log().neg().log().neg();
    let noisy = log_probs.clone() + gumbel;
    let indices = noisy.argmax(1);

    let transaction = Transaction::default()
        .register(log_probs.gather(1, indices.clone()))
        .register(indices)
        .execute();

    (
        transaction[1]
            .iter::<B::IntElem>()
            .map(|x| x.to_usize())
            .collect(),
        transaction[0].to_vec().unwrap(),
    )
}

pub(crate) fn argmax_actions<B: Backend>(output: Tensor<B, 2>) -> Vec<usize> {
    output
        .argmax(1)
        .into_data()
        .iter::<B::IntElem>()
        .map(|x| x.to_usize())
        .collect()
}
