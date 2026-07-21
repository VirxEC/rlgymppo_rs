use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use burn::tensor::cast::ToElement;
use burn::tensor::{Distribution, Transaction};

pub(crate) fn to_mask_tensor_2d<B: Backend>(
    masks: &[Vec<bool>],
    device: &B::Device,
) -> Tensor<B, 2> {
    let shape = [masks.len(), masks[0].len()];
    let data: Vec<f32> = masks
        .iter()
        .flat_map(|mask| mask.iter().map(|&value| if value { 1.0 } else { 0.0 }))
        .collect();
    Tensor::from_data(TensorData::new(data, shape), device)
}

pub(crate) fn to_mask_tensor_2d_indexed<B: Backend>(
    masks: &[Vec<bool>],
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let columns = masks[indices[0]].len();
    let mut data = Vec::with_capacity(indices.len() * columns);
    for &index in indices {
        data.extend(
            masks[index]
                .iter()
                .map(|&value| if value { 1.0 } else { 0.0 }),
        );
    }
    Tensor::from_data(TensorData::new(data, [indices.len(), columns]), device)
}

pub(crate) fn to_state_tensor_2d<B: Backend>(
    state: &[Vec<f32>],
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut data: Vec<f32> = Vec::with_capacity(state.len() * state[0].len());
    for row in state {
        data.extend(row);
    }
    Tensor::from_data(TensorData::new(data, [state.len(), state[0].len()]), device)
}

pub(crate) fn to_state_tensor_2d_indexed<B: Backend>(
    state: &[Vec<f32>],
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let columns = state[indices[0]].len();
    let mut data: Vec<f32> = Vec::with_capacity(indices.len() * columns);
    for &index in indices {
        data.extend(&state[index]);
    }
    Tensor::from_data(TensorData::new(data, [indices.len(), columns]), device)
}

pub(crate) struct SampledActions<B: Backend> {
    pub actions: Tensor<B, 2, Int>,
    pub log_probs: Tensor<B, 2>,
}

pub(crate) fn sample_actions_from_logits_tensor<B: Backend>(
    logits: Tensor<B, 2>,
    device: &B::Device,
) -> SampledActions<B> {
    let shape = logits.shape();
    let log_probs = log_softmax(logits.clone(), 1);
    let uniform = Tensor::<B, 2>::random(shape, Distribution::Default, device);
    let gumbel = uniform.log().neg().log().neg();
    let actions = (logits + gumbel).argmax(1);
    let log_probs = log_probs.gather(1, actions.clone());

    SampledActions { actions, log_probs }
}

pub(crate) fn sampled_actions_to_vec<B: Backend>(
    sampled: SampledActions<B>,
) -> (Vec<usize>, Vec<f32>) {
    let transaction = Transaction::default()
        .register(sampled.log_probs)
        .register(sampled.actions)
        .execute();

    (
        transaction[1]
            .iter::<B::IntElem>()
            .map(|value| value.to_usize())
            .collect(),
        transaction[0].to_vec().unwrap(),
    )
}

pub(crate) fn sample_actions_from_logits<B: Backend>(
    logits: Tensor<B, 2>,
    device: &B::Device,
) -> (Vec<usize>, Vec<f32>) {
    sampled_actions_to_vec(sample_actions_from_logits_tensor(logits, device))
}

pub(crate) fn argmax_actions<B: Backend>(output: Tensor<B, 2>) -> Vec<usize> {
    output
        .argmax(1)
        .into_data()
        .iter::<B::IntElem>()
        .map(|value| value.to_usize())
        .collect()
}
