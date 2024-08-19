use tch::{
    kind::INT64_CPU,
    IndexOp, Tensor,
};

#[derive(Default)]
pub struct TrajectoryTensors {
    states: Tensor,
    actions: Tensor,
    log_probs: Tensor,
    rewards: Tensor,
    next_states: Tensor,
    dones: Tensor,
    truncateds: Tensor,
}

impl TrajectoryTensors {
    pub const NUM_TENSORS: usize = 7;

    pub fn new(
        states: Tensor,
        actions: Tensor,
        log_probs: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
        truncateds: Tensor,
    ) -> Self {
        Self {
            states,
            actions,
            log_probs,
            rewards,
            next_states,
            dones,
            truncateds,
        }
    }

    fn array_of_refs(&self) -> [&Tensor; Self::NUM_TENSORS] {
        [
            &self.states,
            &self.actions,
            &self.log_probs,
            &self.rewards,
            &self.next_states,
            &self.dones,
            &self.truncateds,
        ]
    }

    fn array_of_muts(&mut self) -> [&mut Tensor; Self::NUM_TENSORS] {
        [
            &mut self.states,
            &mut self.actions,
            &mut self.log_probs,
            &mut self.rewards,
            &mut self.next_states,
            &mut self.dones,
            &mut self.truncateds,
        ]
    }
}

/// A container for the timestep data of a specific agent
/// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/batched_agents/batched_trajectory.py
/// Unlike rlgym-ppo, this has a capacity allocation system like Vec
/// This class is designed to append single steps or merge multiple trajectories as fast as possible
#[derive(Default)]
pub struct Trajectory {
    tensors: TrajectoryTensors,
    capacity: usize,
    length: usize,
}

impl Trajectory {
    fn double_reserve(&mut self) {
        if self.capacity > 0 {
            for t in self.tensors.array_of_muts() {
                *t = t.repeat([2, 1]);
            }

            self.capacity *= 2;
        }
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn mark_if_truncated(&mut self) {
        let last_is_done = self
            .tensors
            .dones
            .i((self.length - 1) as i64)
            .iter::<f64>()
            .unwrap()
            .next()
            .unwrap();
        let _ = self
            .tensors
            .truncateds
            .i((self.length - 1) as i64)
            .fill_(1. - last_is_done);
    }

    pub fn append_single_step(&mut self, mut step: TrajectoryTensors) {
        if self.length == 0 {
            for (current, new) in self
                .tensors
                .array_of_muts()
                .into_iter()
                .zip(step.array_of_muts())
            {
                *current = new.unsqueeze(0);
            }

            self.length = 1;
            self.capacity = 1;
        } else {
            if self.length == self.capacity {
                self.double_reserve();
            }

            let index_tensor = Tensor::scalar_tensor(self.length as i64, INT64_CPU);
            for (current, new) in self
                .tensors
                .array_of_muts()
                .into_iter()
                .zip(step.array_of_muts())
            {
                let _ = current.index_copy_(0, &index_tensor, &new.unsqueeze(0));
            }

            self.length += 1;
        }
    }

    pub fn multi_append(&mut self, others: Vec<Trajectory>) {
        let already_have_data = self.length != 0;

        for i in 0..TrajectoryTensors::NUM_TENSORS {
            let mut cat_list = Vec::with_capacity(others.len());

            if already_have_data {
                cat_list.push(self.tensors.array_of_refs()[i].shallow_clone());
            }

            for other_traj in &others {
                let sliced_data =
                    other_traj.tensors.array_of_refs()[i].slice(0, 0, other_traj.length as i64, 1);
                cat_list.push(sliced_data);
            }

            *self.tensors.array_of_muts()[i] = Tensor::cat(&cat_list, 0);
        }

        let new_length = self.tensors.states.size()[0] as usize;
        self.length = new_length;
        self.capacity = new_length;
    }

    pub fn clear(&mut self) {}
}
