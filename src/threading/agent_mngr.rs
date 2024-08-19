use super::{
    agent::{AgentConfig, AgentController, AgentControls},
    trajectory::Trajectory,
};
use crate::{
    ppo::discrete::DiscretePolicy,
    util::{avg_tracker::AvgTracker, report::Report},
};
use rlgym_rs::{Action, Env, Obs, Reward, StateSetter, Terminal, Truncate};
use std::{
    sync::{atomic::Ordering, Arc},
    thread::sleep,
    time::Duration,
};
use tch::Device;

pub struct AgentManager {
    policy: Arc<DiscretePolicy>,
    agents: Vec<AgentController>,
    agent_config: AgentConfig,
    agent_controls: Arc<AgentControls>,
    max_collect: u64,
    collect_during_training: bool,
}

impl AgentManager {
    pub fn new(
        policy: Arc<DiscretePolicy>,
        max_collect: u64,
        deterministic: bool,
        collect_during_training: bool,
        controls_update_frequency: u64,
        device: Device,
    ) -> Self {
        Self {
            policy,
            max_collect,
            collect_during_training,
            agents: Vec::new(),
            agent_config: AgentConfig {
                deterministic,
                device,
                controls_update_frequency,
                max_steps: 0,
                num_games: 0,
            },
            agent_controls: Arc::default(),
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
                self.agent_controls.clone(),
                self.policy.clone(),
                create_env_fn.clone(),
                i,
            );
            self.agents.push(agent);
        }
    }

    pub fn start(&self) {
        self.agent_controls
            .should_run
            .store(true, Ordering::Relaxed);
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

            sleep(Duration::from_millis(1));
        }

        if !self.agent_config.deterministic && !self.collect_during_training {
            self.agent_controls.paused.store(true, Ordering::Relaxed);
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

        debug_assert_eq!(
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

        // policy is only updated when deterministic is false
        // so we don't need to check in the if statement
        if !self.collect_during_training {
            self.agent_controls.paused.store(false, Ordering::Relaxed);
        }
    }

    pub fn get_metrics(&mut self, report: &mut Report) {
        let mut avg_step_rew = AvgTracker::default();
        let mut avg_ep_rew = AvgTracker::default();

        for agent in &self.agents {
            let mut agent_metrics = agent.get_metrics();

            avg_step_rew += agent_metrics.avg_steps_reward;
            avg_ep_rew += agent_metrics.avg_episode_reward;
            *report += &agent_metrics.report;

            agent_metrics.reset();
        }

        report["Average Step Reward"] = avg_step_rew.get();
        report["Average Episode Reward"] = avg_ep_rew.get();
        report["Policy infer time"] /= self.agents.len() as f64;
        report["Env step time"] =
            (report["Env step time"] + report["Trajectory append time"]) / self.agents.len() as f64;
        report.remove("Trajectory append time");
    }

    pub fn stop(&mut self) {
        println!("Stopping agents...");
        self.agent_controls
            .should_run
            .store(false, Ordering::Relaxed);

        for agent in self.agents.drain(..) {
            agent.wait_for_close();
        }
    }
}
