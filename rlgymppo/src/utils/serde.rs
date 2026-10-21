use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use burn::prelude::*;

use super::running_stat::Stats;
use crate::agent::Ppo;
use crate::agent::model::Actic;

/// Save a model checkpoint (model weights + training stats).
pub fn save_model<P: AsRef<Path>>(
    model: Actic,
    running_stats: &Stats,
    base_folder: P,
    limit: Option<usize>,
) -> PathBuf {
    #[cfg(not(feature = "tui"))]
    println!("Saving model...");

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let save_folder = base_folder.as_ref().join(timestamp.to_string());
    fs::create_dir_all(&save_folder).unwrap();

    // Save each component to its own file.
    model
        .actor
        .save_file(save_folder.join("actor.bpk"))
        .unwrap();
    model
        .critic
        .save_file(save_folder.join("critic.bpk"))
        .unwrap();
    if let Some(ref head) = model.shared_head {
        head.clone()
            .save_file(save_folder.join("shared_head.bpk"))
            .unwrap();
    }

    let toml_str = toml::to_string(running_stats).unwrap();
    fs::write(save_folder.join("stats.toml"), toml_str).unwrap();

    #[cfg(not(feature = "tui"))]
    println!("Saved model to: {save_folder:?}");

    if let Some(limit) = limit {
        let Ok(folders) = save_folder.parent().unwrap().read_dir() else {
            #[cfg(not(feature = "tui"))]
            println!("Failed to read directory: {save_folder:?}");
            return save_folder;
        };

        let mut folders: Vec<_> = folders
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().is_dir())
            .collect();
        if folders.len() <= limit {
            return save_folder;
        }

        folders.sort_by_key(|entry| entry.file_name());

        let need_removal = folders.len() - limit;
        for folder in folders.into_iter().take(need_removal) {
            let oldest = folder.path();
            fs::remove_dir_all(&oldest).unwrap();
            #[cfg(not(feature = "tui"))]
            println!("Removed old model folder: {oldest:?}");
        }
    }

    save_folder
}

/// Save a full checkpoint: model weights, training stats, and optimizer states.
/// The model uses a runtime device. Training devices enable autodiff.
pub fn save_checkpoint<P: AsRef<Path>>(
    model: Actic,
    ppo: &Ppo,
    running_stats: &Stats,
    base_folder: P,
    limit: Option<usize>,
) -> PathBuf {
    let folder = save_model(model, running_stats, &base_folder, limit);
    ppo.save_optimizers(&folder);
    ppo.save_regen_snapshot(&folder);
    folder
}

/// Find the latest checkpoint folder by timestamp.
pub fn latest_checkpoint_folder(base_folder: &Path) -> Option<PathBuf> {
    let Ok(folders) = base_folder.read_dir() else {
        return None;
    };

    folders
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if !path.is_dir() {
                return None;
            }
            let timestep = entry.file_name().to_str()?.parse::<u64>().ok()?;
            Some((timestep, path))
        })
        .max_by_key(|(timestep, _)| *timestep)
        .map(|(_, path)| path)
}

/// Resolve the folder containing a saved model's component files: `path`
/// itself when it directly contains an `actor.bpk`, otherwise the latest
/// timestamped checkpoint subfolder (see [`latest_checkpoint_folder`]).
pub fn resolve_model_folder(path: &Path) -> Option<PathBuf> {
    if path.join("actor.bpk").is_file() {
        Some(path.to_owned())
    } else if path.join("actor.mpk.gz").is_file() || path.join("actor.mpk").is_file() {
        panic!(
            "Found legacy MessagePack checkpoint in {}. Burn 0.22 cannot load .mpk files; convert it with Burn 0.21 before loading.",
            path.display()
        );
    } else {
        latest_checkpoint_folder(path)
    }
}

pub fn load_latest_model<P: AsRef<Path>>(
    model: Actic,
    base_folder: P,
    device: &Device,
) -> (Actic, Stats) {
    let base_folder = base_folder.as_ref();
    let Some(latest_folder) = resolve_model_folder(base_folder) else {
        #[cfg(not(feature = "tui"))]
        println!("No valid folders found in: {:?}", base_folder.display());
        return (model, Stats::default());
    };

    load_model(model, latest_folder, device)
}

pub fn load_model<P: AsRef<Path>>(model: Actic, file_path: P, _device: &Device) -> (Actic, Stats) {
    let path = file_path.as_ref();
    assert!(path.exists(), "Model path does not exist: {path:?}");
    for component in ["actor", "critic", "shared_head"] {
        for extension in ["mpk.gz", "mpk"] {
            let legacy = path.join(format!("{component}.{extension}"));
            if legacy.is_file() && !path.join(format!("{component}.bpk")).is_file() {
                panic!(
                    "Found legacy MessagePack checkpoint {}. Burn 0.22 cannot load .mpk files; convert it with Burn 0.21 before loading.",
                    legacy.display()
                );
            }
        }
    }

    // Load each component from its own file.
    let actor = model
        .actor
        .try_load_file(path.join("actor.bpk"))
        .unwrap_or_else(|error| {
            panic!(
                "Failed to load actor checkpoint in {}: {error}",
                path.display()
            )
        });
    let critic = model
        .critic
        .try_load_file(path.join("critic.bpk"))
        .unwrap_or_else(|error| {
            panic!(
                "Failed to load critic checkpoint in {}: {error}",
                path.display()
            )
        });
    let shared_head = model.shared_head.map(|head| {
        let head_path = path.join("shared_head.bpk");
        head.clone()
            .try_load_file(&head_path)
            .unwrap_or_else(|error| {
                panic!(
                    "Failed to load shared-head checkpoint {}: {error}",
                    head_path.display()
                )
            })
    });

    let toml_str = fs::read_to_string(path.join("stats.toml")).unwrap();
    let stats: Stats = toml::from_str(&toml_str).unwrap();

    #[cfg(not(feature = "tui"))]
    println!("Loaded model from: {path:?}");

    (
        Actic {
            actor,
            critic,
            shared_head,
        },
        stats,
    )
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[test]
    fn regression_checkpoint_discovery_ignores_policy_version_directory() {
        let root = std::env::temp_dir().join(format!(
            "rlgymppo-serde-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("policy_versions")).unwrap();
        fs::create_dir_all(root.join("20")).unwrap();
        fs::create_dir_all(root.join("100")).unwrap();
        fs::write(root.join("not-a-checkpoint"), []).unwrap();

        assert_eq!(latest_checkpoint_folder(&root), Some(root.join("100")));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn regression_resolve_model_folder_prefers_direct_checkpoint() {
        let root = std::env::temp_dir().join(format!(
            "rlgymppo-serde-resolve-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("direct")).unwrap();
        fs::write(root.join("direct").join("actor.bpk"), []).unwrap();
        fs::create_dir_all(root.join("10")).unwrap();
        fs::create_dir_all(root.join("20")).unwrap();

        // A folder that directly contains model files wins over timestamp
        // discovery from the parent.
        assert_eq!(
            resolve_model_folder(&root.join("direct")),
            Some(root.join("direct"))
        );
        assert_eq!(resolve_model_folder(&root), Some(root.join("20")));

        fs::remove_dir_all(root).unwrap();
    }
}
