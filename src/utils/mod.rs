mod avg_tracker;
mod report;

pub use avg_tracker::AvgTracker;
pub use report::{Report, Reportable};

use burn::LearningRate;
use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData, Transaction};
use rand::{
    Rng,
    distr::{Distribution, weighted::WeightedIndex},
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

pub(crate) fn sample_actions_from_tensor<B: Backend, R: Rng>(
    output: Tensor<B, 2>,
    rng: &mut R,
) -> Vec<usize> {
    let num_actions = output.shape().dims[0];

    let mut transaction = Transaction::default();
    for data in output.iter_dim(0) {
        transaction = transaction.register(data);
    }

    let mut actions = Vec::with_capacity(num_actions);
    for data in transaction.execute() {
        let prob = data.into_vec::<f32>().unwrap();
        let dist = WeightedIndex::new(prob).unwrap();

        actions.push(dist.sample(rng));
    }

    debug_assert_eq!(actions.len(), num_actions);
    actions
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
