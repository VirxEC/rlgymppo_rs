use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

/// Appends one JSON object per metric report to a local `.jsonl` file.
///
/// The metrics actor owns this sink; it is cheap (buffered + flush per line)
/// and lives on the actor thread, so it never stalls collection or training.
pub struct MetricsJsonlSink {
    writer: Mutex<BufWriter<std::fs::File>>,
}

impl MetricsJsonlSink {
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())?;
        Ok(Self {
            writer: Mutex::new(BufWriter::new(file)),
        })
    }

    pub fn write_line(&self, flat: &HashMap<String, f64>) -> std::io::Result<()> {
        let mut writer = self.writer.lock().unwrap();
        serde_json::to_writer(&mut *writer, flat)?;
        writer.write_all(b"\n")?;
        writer.flush()
    }
}
