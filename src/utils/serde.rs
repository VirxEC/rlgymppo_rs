use std::{
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use burn::{
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkGzFileRecorder},
};

use super::running_stat::Stats;
use crate::agent::model::Actic;

pub fn save_model<B: Backend, P: AsRef<Path>>(
    model: Actic<B>,
    running_stats: &Stats,
    base_folder: P,
    limit: Option<usize>,
) {
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

    println!("Saved model to: {save_folder:?}");

    if let Some(limit) = limit {
        let Ok(folders) = save_folder.parent().unwrap().read_dir() else {
            println!("Failed to read directory: {save_folder:?}");
            return;
        };

        let mut folders: Vec<_> = folders
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().is_dir())
            .collect();
        if folders.len() <= limit {
            return;
        }

        folders.sort_by_key(|entry| entry.file_name());

        let need_removal = folders.len() - limit;
        for folder in folders.into_iter().take(need_removal) {
            let oldest = folder.path();
            fs::remove_dir_all(&oldest).unwrap();
            println!("Removed old model folder: {oldest:?}");
        }
    }
}

pub fn load_latest_model<B: Backend, P: AsRef<Path>>(
    model: Actic<B>,
    base_folder: P,
    device: &B::Device,
) -> (Actic<B>, Stats) {
    let base_folder = base_folder.as_ref();
    let Ok(folders) = base_folder.read_dir() else {
        println!("Failed to read directory: {:?}", base_folder.display());
        return (model, Stats::default());
    };

    let Some(latest_folder) = folders
        .filter_map(|entry| entry.ok())
        .max_by_key(|entry| entry.file_name().to_str().unwrap().parse::<u64>().ok())
    else {
        println!("No valid folders found in: {:?}", base_folder.display());
        return (model, Stats::default());
    };

    load_model(model, latest_folder.path(), device)
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
