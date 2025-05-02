use crate::agent::model::Actic;
use burn::{
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkGzFileRecorder},
};
use std::{
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

pub fn save_model<B: Backend, P: AsRef<Path>>(
    model: Actic<B>,
    base_folder: P,
    limit: Option<usize>,
) {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let base_folder = base_folder.as_ref().join(timestamp.to_string());
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file(base_folder.join("model"), &recorder)
        .unwrap();

    println!("Saved model to: {base_folder:?}");

    if let Some(limit) = limit {
        // cap the number of saved checkpoints at `limit`
        let Ok(folders) = base_folder.parent().unwrap().read_dir() else {
            println!("Failed to read directory: {base_folder:?}");
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
) -> Actic<B> {
    let base_folder = base_folder.as_ref();
    let Ok(folders) = base_folder.read_dir() else {
        println!("Failed to read directory: {:?}", base_folder.display());
        return model;
    };

    let Some(latest_folder) = folders
        .filter_map(|entry| entry.ok())
        .max_by_key(|entry| entry.file_name().to_str().unwrap().parse::<u64>().ok())
    else {
        println!("No valid folders found in: {:?}", base_folder.display());
        return model;
    };

    load_model(model, latest_folder.path(), device)
}

pub fn load_model<B: Backend, P: AsRef<Path>>(
    model: Actic<B>,
    file_path: P,
    device: &B::Device,
) -> Actic<B> {
    let path = file_path.as_ref();
    assert!(path.exists(), "Model path does not exist: {path:?}");
    println!("Loading model from: {path:?}");

    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    model
        .load_file(path.join("model"), &recorder, device)
        .unwrap()
}
