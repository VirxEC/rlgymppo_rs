mod avg_tracker;
mod report;
pub(crate) mod running_stat;
pub(crate) mod serde;

pub use avg_tracker::AvgTracker;
pub use report::{Report, Reportable};

use burn::{
    LearningRate,
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
    prelude::*,
    tensor::{Distribution, Transaction, backend::AutodiffBackend, cast::ToElement},
};

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
    let shape = action_probs.shape().dims;
    let log_probs = action_probs.log();

    let u = Tensor::<B, 2>::random(shape, Distribution::Default, device);
    let gumbel = u.log().neg().log().neg();
    let noisy = log_probs.clone() + gumbel;
    let indices = noisy.argmax(1);

    let transation = Transaction::default()
        .register(log_probs.gather(1, indices.clone()))
        .register(indices)
        .execute();

    (
        transation[1]
            .iter::<B::IntElem>()
            .map(|x| x.to_usize())
            .collect(),
        transation[0].to_vec().unwrap(),
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

pub(crate) fn elementwise_min<B: Backend, const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
) -> Tensor<B, D> {
    let rhs_lower = rhs.clone().lower(lhs.clone());
    lhs.clone().mask_where(rhs_lower, rhs.clone())
}

pub(crate) fn update_parameters<B: AutodiffBackend, M: AutodiffModule<B>>(
    loss: Tensor<B, 1>,
    module: M,
    optimizer: &mut impl Optimizer<M, B>,
    learning_rate: LearningRate,
) -> M {
    let gradients = loss.backward();
    let gradient_params = GradientsParams::from_grads(gradients, &module);
    optimizer.step(learning_rate, module, gradient_params)
}
