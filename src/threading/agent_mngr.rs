use super::{
    agent::{AgentConfig, AgentController},
    trajectory::Trajectory,
};
use crate::ppo::discrete::DiscretePolicy;
use rlgym_rs::{Action, Env, Obs, Reward, StateSetter, Terminal, Truncate};
use std::{sync::Arc, thread::sleep, time::Duration};
use tch::Device;

pub struct AgentManager {
    policy: Arc<DiscretePolicy>,
    agents: Vec<AgentController>,
    agent_config: AgentConfig,
    max_collect: u64,
}

impl AgentManager {
    pub fn new(
        policy: Arc<DiscretePolicy>,
        max_collect: u64,
        deterministic: bool,
        device: Device,
    ) -> Self {
        Self {
            policy,
            max_collect,
            agents: Vec::new(),
            agent_config: AgentConfig {
                deterministic,
                device,
                max_steps: 0,
                num_games: 0,
            },
        }
    }

    pub fn create_agents<F, SS, OBS, ACT, REW, TERM, TRUNC, SI>(
        &mut self,
        create_env_fn: F,
        amount: usize,
        games_per_agent: usize,
    ) where
        F: Fn() -> Env<SS, OBS, ACT, REW, TERM, TRUNC, SI> + Send + Clone + 'static,
        SS: StateSetter<SI>,
        OBS: Obs<SI>,
        ACT: Action<SI, Input = Vec<i32>>,
        REW: Reward<SI>,
        TERM: Terminal<SI>,
        TRUNC: Truncate<SI>,
    {
        self.agent_config.num_games = games_per_agent;
        self.agent_config.max_steps = self.max_collect / amount as u64;

        for i in 0..amount {
            let agent = AgentController::new(
                self.agent_config.clone(),
                self.policy.clone(),
                create_env_fn.clone(),
                i,
            );
            self.agents.push(agent);
        }
    }

    pub fn start(&self) {
        for agent in &self.agents {
            agent.start();
        }
    }

    pub fn collect_timesteps(&self, timesteps: u64) -> Trajectory {
        // wait in this loop until agents have enough timesteps
        loop {
            let total_steps: u64 = self
                .agents
                .iter()
                .map(AgentController::get_num_timesteps)
                .sum();

            if total_steps >= timesteps {
                break;
            }

            sleep(Duration::from_millis(2));
        }

        let mut result = Trajectory::default();
        let mut total_timesteps = 0;

        {
            let mut game_trajectories = Vec::new();
            for agent in &self.agents {
                let mut trajectories = agent.get_trajectories();

                for traj_set in &mut *trajectories {
                    for traj in traj_set {
                        if traj.is_empty() {
                            continue;
                        }

                        // If the last timestep is not a done, mark it as truncated
                        // The GAE needs to know when the environment state stops being continuous
                        // This happens either because the environment reset (i.e. goal scored), called "done",
                        //	or the data got cut short, called "truncated"

                        traj.mark_if_truncated();

                        let existing_traj = std::mem::take(traj);

                        total_timesteps += existing_traj.len();
                        game_trajectories.push(existing_traj);
                    }
                }

                agent.reset_timesteps();
            }

            result.multi_append(game_trajectories);
        }

        debug_assert_ne!(
            result.len(),
            total_timesteps,
            "Trajectory length does not match total timesteps"
        );

        result
    }

    pub fn update_policy(&mut self, policy: Arc<DiscretePolicy>) {
        for agent in &mut self.agents {
            agent.update_policy(policy.clone());
        }
    }

    pub fn stop(&mut self) {
        println!("Stopping agents...");
        for agent in &self.agents {
            agent.request_stop();
        }

        for agent in self.agents.drain(..) {
            agent.wait_for_close();
        }
    }
}
