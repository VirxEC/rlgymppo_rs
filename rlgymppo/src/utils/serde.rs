use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use burn::{
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkGzFileRecorder},
    tensor::backend::AutodiffBackend,
};

use super::running_stat::Stats;
use crate::agent::{Ppo, model::Actic};

/// Save a model checkpoint (model weights + training stats).
pub fn save_model<B: Backend, P: AsRef<Path>>(
    model: Actic<B>,
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

    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

    // Save each component to its own file.
    model
        .actor
        .save_file(save_folder.join("actor"), &recorder)
        .unwrap();
    model
        .critic
        .save_file(save_folder.join("critic"), &recorder)
        .unwrap();
    if let Some(ref head) = model.shared_head {
        head.clone()
            .save_file(save_folder.join("shared_head"), &recorder)
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
/// The model may use the inner backend (without gradients) while the Ppo
/// uses the autodiff backend.
pub fn save_checkpoint<BAutodiff: AutodiffBackend, P: AsRef<Path>>(
    model: Actic<BAutodiff::InnerBackend>,
    ppo: &Ppo<BAutodiff>,
    running_stats: &Stats,
    base_folder: P,
    limit: Option<usize>,
) -> PathBuf {
    let folder = save_model(model, running_stats, &base_folder, limit);
    ppo.save_optimizers(&folder);
    folder
}

/// Find the latest checkpoint folder by timestamp.
pub fn latest_checkpoint_folder(base_folder: &Path) -> Option<PathBuf> {
    let Ok(folders) = base_folder.read_dir() else {
        return None;
    };

    folders
        .filter_map(|entry| entry.ok())
        .max_by_key(|entry| entry.file_name().to_str().unwrap().parse::<u64>().ok())
        .map(|entry| entry.path())
}

pub fn load_latest_model<B: Backend, P: AsRef<Path>>(
    model: Actic<B>,
    base_folder: P,
    device: &B::Device,
) -> (Actic<B>, Stats) {
    let base_folder = base_folder.as_ref();
    let Some(latest_folder) = latest_checkpoint_folder(base_folder) else {
        #[cfg(not(feature = "tui"))]
        println!("No valid folders found in: {:?}", base_folder.display());
        return (model, Stats::default());
    };

    load_model(model, latest_folder, device)
}

pub fn load_model<B: Backend, P: AsRef<Path>>(
    model: Actic<B>,
    file_path: P,
    device: &B::Device,
) -> (Actic<B>, Stats) {
    let path = file_path.as_ref();
    assert!(path.exists(), "Model path does not exist: {path:?}");

    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

    // Load each component from its own file.
    let actor = model
        .actor
        .load_file(path.join("actor"), &recorder, device)
        .unwrap();
    let critic = model
        .critic
        .load_file(path.join("critic"), &recorder, device)
        .unwrap();
    let shared_head = model.shared_head.map(|head| {
        let head_path = path.join("shared_head");
        if head_path.exists() {
            head.load_file(head_path, &recorder, device).unwrap()
        } else {
            head
        }
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
