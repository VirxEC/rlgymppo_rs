use fastrand::Rng;
use tch::{Device, Kind, Scalar, Tensor};

pub struct SampleSet {
    pub actions: Tensor,
    pub log_probs: Tensor,
    pub states: Tensor,
    pub values: Tensor,
    pub advantages: Tensor,
}

#[derive(Default)]
pub struct ExperienceTensors {
    pub states: Tensor,
    pub actions: Tensor,
    pub log_probs: Tensor,
    pub rewards: Tensor,
    pub next_states: Tensor,
    pub dones: Tensor,
    pub truncateds: Tensor,
    pub values: Tensor,
    pub advantages: Tensor,
}

impl ExperienceTensors {
    fn array_of_muts(&mut self) -> [&mut Tensor; 9] {
        [
            &mut self.states,
            &mut self.actions,
            &mut self.log_probs,
            &mut self.rewards,
            &mut self.next_states,
            &mut self.dones,
            &mut self.truncateds,
            &mut self.values,
            &mut self.advantages,
        ]
    }
}

pub struct ExperienceBuffer {
    device: Device,
    seed: u64,
    data: ExperienceTensors,
    cur_size: i64,
    max_size: i64,
    rng: Rng,
}

impl ExperienceBuffer {
    pub fn new(max_size: u64, seed: u64, device: Device) -> Self {
        Self {
            device,
            seed,
            data: Default::default(),
            cur_size: 0,
            max_size: max_size as i64,
            rng: Rng::with_seed(seed),
        }
    }

    pub fn submit_experience(&mut self, mut new_data: ExperienceTensors) {
        let _no_grad = tch::no_grad_guard();

        let empty = self.cur_size == 0;

        for (curr_ten, new_ten) in self
            .data
            .array_of_muts()
            .into_iter()
            .zip(new_data.array_of_muts().into_iter())
        {
            let mut new_size = new_ten.size()[0];

            if new_size > self.max_size {
                *new_ten = new_ten.slice(0, new_size - self.max_size, new_size, 1);
                new_size = self.max_size;
            }

            let overflow = ((self.cur_size + new_size) - self.max_size).max(0);
            let start_idx = self.cur_size - overflow;
            let end_idx = self.cur_size + new_size - overflow;

            if empty {
                // Initalize tensor

                // Make zero tensor of target size
                let sizes = new_ten.size();
                let mut new_sizes = sizes.to_vec();
                new_sizes[0] = self.max_size;
                *curr_ten = Tensor::zeros(&new_sizes, (Kind::Float, self.device));

                // Make ourTen NAN, such that it is obvious if uninitialized data is being used
                let _ = curr_ten.f_add_scalar_(Scalar::float(f64::NAN)).unwrap();

                assert_eq!(curr_ten.size()[0], self.max_size);
            } else {
                // We already have data
                if overflow > 0 {
                    let from_data = curr_ten
                        .slice(0, overflow, self.cur_size, 1)
                        .shallow_clone();
                    let mut to_view = curr_ten.slice(0, 0, self.cur_size - overflow, 1);
                    to_view.copy_(&from_data);

                    assert!(curr_ten
                        .get(self.cur_size - overflow - 1)
                        .equal(&curr_ten.get(self.cur_size - 1)));
                }
            }

            let mut our_ten_insert_view = curr_ten.slice(0, start_idx, end_idx, 1);
            our_ten_insert_view.copy_(new_ten);
            assert!(curr_ten
                .get(end_idx - 1)
                .equal(&new_ten.get(new_ten.size()[0] - 1)));
        }

        self.cur_size = (self.cur_size + new_data.states.size()[0]).min(self.max_size);
    }

    fn _get_samples(&self, indices: &[i64]) -> SampleSet {
        let t_indices = Tensor::from_slice(indices);

        SampleSet {
            actions: self.data.actions.index_select(0, &t_indices),
            log_probs: self.data.log_probs.index_select(0, &t_indices),
            states: self.data.states.index_select(0, &t_indices),
            values: self.data.values.index_select(0, &t_indices),
            advantages: self.data.advantages.index_select(0, &t_indices),
        }
    }

    pub fn get_all_batches_shuffled(&mut self, batch_size: u64) -> Vec<SampleSet> {
        // Make list of shuffled sample indices
        let mut indices: Vec<i64> = (0..self.cur_size).collect();
        self.rng.shuffle(&mut indices);

        // Get a sample set from each of the batches
        indices
            .chunks_exact(batch_size as usize)
            .map(|chunk| self._get_samples(chunk))
            .collect()
    }
}
