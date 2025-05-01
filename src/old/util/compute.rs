use tch::{Device, Kind, Tensor};

pub trait NonBlockingTransfer {
    fn no_block_to(&self, device: Device) -> Tensor;
}

impl NonBlockingTransfer for Tensor {
    #[inline]
    fn no_block_to(&self, device: Device) -> Tensor {
        self.to_device_(device, Kind::Float, true, false)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gae(
    rews: Vec<f32>,
    dones: Vec<f32>,
    truncated: Vec<f32>,
    values: Vec<f32>,
    gamma: f32,
    lambda: f32,
    return_std: f32,
    clip_range: f32,
) -> (Tensor, Tensor, Vec<f32>) {
    let next_values = values.iter().skip(1).copied().collect::<Vec<f32>>();

    let mut return_scale = 1.0 / return_std;
    if return_scale.is_nan() {
        return_scale = 0.0;
    }

    let mut last_gae_lam = 0.0;
    let n_returns = rews.len();

    let mut adv = vec![0.0; n_returns];
    let mut returns = vec![0.0; n_returns];
    let mut last_return = 0.0;

    for step in (0..n_returns).rev() {
        let not_done = 1.0 - dones[step];
        let not_trunc = 1.0 - truncated[step];

        let norm_rew = if return_std != 0.0 {
            let mut rew = rews[step] * return_scale;

            if clip_range > 0.0 {
                rew = rew.clamp(-clip_range, clip_range)
            }

            rew
        } else {
            rews[step]
        };

        let pred_ret = norm_rew + gamma * next_values[step] * not_done;
        let delta = pred_ret - values[step];
        let ret = rews[step] + last_return * gamma * not_done * not_trunc;
        returns[step] = ret;
        last_return = ret;
        last_gae_lam = delta + gamma * lambda * not_done * not_trunc * last_gae_lam;
        adv[step] = last_gae_lam;
    }

    let out_values_list: Vec<f32> = values.into_iter().zip(&adv).map(|(v, a)| v + a).collect();

    (
        Tensor::from_slice(&adv),
        Tensor::from_slice(&out_values_list),
        returns,
    )
}
