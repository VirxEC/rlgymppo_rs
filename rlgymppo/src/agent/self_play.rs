use std::{collections::HashSet, fs, path::PathBuf};

use burn::{
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkGzFileRecorder},
};
use rand::{Rng, rngs::SmallRng};

use super::model::Actic;

/// Configuration for saving policy versions and training against old
/// versions (self-play).
///
/// When `train_against_old_versions` is enabled, the collector will
/// occasionally use a randomly-chosen saved policy version as the
/// opponent for one team.  This provides a natural curriculum:  the
/// current policy learns to beat increasingly strong versions of itself.
#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    /// Whether to periodically snapshot the current policy.
    /// Automatically enabled when `train_against_old_versions` is true.
    pub save_policy_versions: bool,
    /// Number of timesteps between saving a new policy version.
    pub ts_per_version: u64,
    /// Maximum number of old versions to keep in memory.
    pub max_old_versions: usize,
    /// Whether to occasionally train against old versions (self-play).
    pub train_against_old_versions: bool,
    /// Probability (0.0 – 1.0) that a given training iteration will
    /// use an old version as the opponent.
    pub train_against_old_chance: f32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            save_policy_versions: false,
            ts_per_version: 25_000_000,
            max_old_versions: 32,
            train_against_old_versions: false,
            train_against_old_chance: 0.15,
        }
    }
}

/// A snapshot of the policy network saved at a particular timestep.
/// Uses `Module::clone()` for a deep copy so later mutations of the
/// original network do not affect the stored version.
#[derive(Debug)]
pub struct PolicyVersion<B: Backend> {
    pub timesteps: u64,
    pub model: Actic<B>,
}

impl<B: Backend> PolicyVersion<B> {
    /// Deep-copy the model into a frozen version snapshot.
    pub fn from_model(model: &Actic<B>, timesteps: u64) -> Self {
        // Burn's Module derive provides Clone, which does a true
        // deep-copy (independent parameter buffers).
        Self {
            timesteps,
            model: model.clone(),
        }
    }
}

/// Manages a sliding window of saved policy versions for self-play.
pub struct VersionManager<B: Backend> {
    pub versions: Vec<PolicyVersion<B>>,
    config: SelfPlayConfig,
    #[allow(dead_code)]
    save_folder: PathBuf,
}

impl<B: Backend> VersionManager<B> {
    pub fn new(save_folder: PathBuf, config: SelfPlayConfig) -> Self {
        Self {
            versions: Vec::new(),
            config,
            save_folder,
        }
    }

    /// Whether any saved versions exist.
    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }

    /// Return a uniformly-random index into `self.versions`.
    pub fn random_index(&self, rng: &mut SmallRng) -> usize {
        rng.next_u32() as usize % self.versions.len()
    }

    /// Called after every training iteration.  Snapshots a new version
    /// when crossing a `ts_per_version` boundary, and prunes the oldest
    /// version when the sliding window is full.
    pub fn on_iteration(&mut self, model: &Actic<B>, cur_timesteps: u64, prev_timesteps: u64) {
        if !self.config.save_policy_versions {
            return;
        }

        let crossed = cur_timesteps / self.config.ts_per_version
            > prev_timesteps / self.config.ts_per_version;
        if crossed || prev_timesteps == 0 {
            #[cfg(not(feature = "tui"))]
            println!(" > Saving policy version at {cur_timesteps} ts ...");

            self.versions
                .push(PolicyVersion::from_model(model, cur_timesteps));

            while self.versions.len() > self.config.max_old_versions {
                self.versions.remove(0);
            }

            #[cfg(not(feature = "tui"))]
            println!(
                " > Saved policy version at {cur_timesteps} ts ({} versions now)",
                self.versions.len()
            );
        }
    }

    // ── Disk persistence ─────────────────────────────────────────

    /// Persist all in-memory versions to disk inside
    /// `save_folder/<timestep>/`.  Uses Burn's native recorder
    /// (`.mpk.gz`), identical to how checkpoints are saved.
    ///
    /// Orphaned directories on disk (those whose timestep is no
    /// longer in the in-memory window) are removed.
    pub fn save_versions(&self) {
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

        fs::create_dir_all(&self.save_folder).ok();

        let in_memory: HashSet<u64> = self.versions.iter().map(|v| v.timesteps).collect();

        // Remove saved directories for versions that were pruned.
        if let Ok(entries) = fs::read_dir(&self.save_folder) {
            for entry in entries.flatten() {
                if !entry.file_type().is_ok_and(|t| t.is_dir()) {
                    continue;
                }
                if let Some(name) = entry.file_name().to_str()
                    && let Ok(ts) = name.parse::<u64>()
                    && !in_memory.contains(&ts)
                {
                    let _ = fs::remove_dir_all(entry.path());
                    #[cfg(not(feature = "tui"))]
                    println!(" > Removed orphaned policy version {ts}");
                }
            }
        }

        // Save each version.
        for version in &self.versions {
            let path = self.save_folder.join(version.timesteps.to_string());
            if path.exists() {
                continue; // already on disk
            }
            fs::create_dir_all(&path).unwrap();

            version
                .model
                .actor
                .clone()
                .save_file(path.join("actor"), &recorder)
                .unwrap();
            version
                .model
                .critic
                .clone()
                .save_file(path.join("critic"), &recorder)
                .unwrap();
            if let Some(ref head) = version.model.shared_head {
                head.clone()
                    .save_file(path.join("shared_head"), &recorder)
                    .unwrap();
            }

            #[cfg(not(feature = "tui"))]
            println!(" > Persisted policy version {} to disk", version.timesteps);
        }
    }

    /// Load previously-saved versions from `save_folder`.  A template
    /// model is required to create new network instances (the weights
    /// are replaced from disk).  Versions whose timestep exceeds
    /// `cur_timesteps` are skipped to avoid loading checkpoints that
    /// are newer than the current model.
    pub fn load_versions(&mut self, template: &Actic<B>, device: &B::Device, cur_timesteps: u64) {
        self.versions.clear();

        if !self.save_folder.exists() {
            return;
        }

        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

        // Discover numbered directories.
        let mut timesteps: Vec<u64> = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.save_folder) {
            for entry in entries.flatten() {
                if !entry.file_type().is_ok_and(|t| t.is_dir()) {
                    continue;
                }
                if let Some(name) = entry.file_name().to_str()
                    && let Ok(ts) = name.parse::<u64>()
                {
                    if ts <= cur_timesteps {
                        timesteps.push(ts);
                    } else {
                        #[cfg(not(feature = "tui"))]
                        eprintln!(
                            " > Warning: policy version {ts} is newer than current model \
                             ({cur_timesteps}) – skipping"
                        );
                    }
                }
            }
        }

        timesteps.sort();

        for ts in timesteps {
            let path = self.save_folder.join(ts.to_string());

            // Destructure the template so we can move individual fields.
            let Actic {
                actor: t_actor,
                critic: t_critic,
                shared_head: t_head,
            } = template.clone();

            // `load_file` consumes the module, so we clone a backup for
            // the fallback path.  Cloning is cheap relative to I/O.
            let actor = t_actor
                .clone()
                .load_file(path.join("actor"), &recorder, device)
                .unwrap_or(t_actor);
            let critic = t_critic
                .clone()
                .load_file(path.join("critic"), &recorder, device)
                .unwrap_or(t_critic);
            let shared_head = t_head.map(|head| {
                head.clone()
                    .load_file(path.join("shared_head"), &recorder, device)
                    .unwrap_or(head)
            });

            self.versions.push(PolicyVersion {
                timesteps: ts,
                model: Actic {
                    actor,
                    critic,
                    shared_head,
                },
            });
        }

        // Enforce the sliding window.
        while self.versions.len() > self.config.max_old_versions {
            self.versions.remove(0);
        }

        if !self.versions.is_empty() {
            #[cfg(not(feature = "tui"))]
            println!(
                " > Loaded {} policy version(s) from disk (range {} – {})",
                self.versions.len(),
                self.versions.first().unwrap().timesteps,
                self.versions.last().unwrap().timesteps
            );
        }
    }
}
