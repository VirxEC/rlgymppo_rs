use crate::utils::AvgTracker;
use ahash::AHashMap;
use std::{
    fmt,
    ops::{AddAssign, Index, IndexMut},
};

#[derive(Debug, Clone, Copy)]
pub enum Reportable {
    Float(f64),
    Int(i64),
    Avg(AvgTracker),
}

macro_rules! reportable_from_primitive {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Reportable {
                fn from(val: $t) -> Self {
                    Reportable::Int(val as i64)
                }
            }
        )*
    };
}

reportable_from_primitive!(
    usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128
);

impl From<f64> for Reportable {
    fn from(val: f64) -> Self {
        Reportable::Float(val)
    }
}

impl From<f32> for Reportable {
    fn from(val: f32) -> Self {
        Reportable::Float(val as f64)
    }
}

impl From<AvgTracker> for Reportable {
    fn from(val: AvgTracker) -> Self {
        Reportable::Avg(val)
    }
}

impl Default for Reportable {
    fn default() -> Self {
        Reportable::Float(0.0)
    }
}

impl AddAssign<Reportable> for Reportable {
    fn add_assign(&mut self, other: Reportable) {
        match (self, other) {
            (Reportable::Float(a), Reportable::Float(b)) => *a += b,
            (Reportable::Int(a), Reportable::Int(b)) => *a += b,
            (Reportable::Avg(a), Reportable::Avg(b)) => *a += b,
            (a, b) => *a = b,
        }
    }
}

impl Reportable {
    pub fn as_float(&self) -> f64 {
        match self {
            Reportable::Float(val) => *val,
            _ => unreachable!("Expected Val, got Avg"),
        }
    }

    pub fn as_int(&self) -> i64 {
        match self {
            Reportable::Int(val) => *val,
            _ => unreachable!("Expected Int, got Float"),
        }
    }

    pub fn as_avg(&self) -> AvgTracker {
        match self {
            Reportable::Avg(avg) => *avg,
            _ => unreachable!("Expected Avg, got Val"),
        }
    }

    pub fn as_float_mut(&mut self) -> &mut f64 {
        match self {
            Reportable::Float(val) => val,
            _ => unreachable!("Expected Val, got Avg"),
        }
    }

    pub fn as_int_mut(&mut self) -> &mut i64 {
        match self {
            Reportable::Int(val) => val,
            _ => unreachable!("Expected Int, got Float"),
        }
    }

    pub fn as_avg_mut(&mut self) -> &mut AvgTracker {
        match self {
            Reportable::Avg(avg) => avg,
            _ => unreachable!("Expected Avg, got Val"),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    data: AHashMap<String, Reportable>,
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}Iteration report{}", "-".repeat(25), "-".repeat(25),)?;
        let mut items: Vec<_> = self.data.iter().collect();
        items.sort_unstable_by(|a, b| a.0.cmp(b.0));

        for (key, val) in items {
            match val {
                Reportable::Float(val) => writeln!(f, "\t{key}: {val}")?,
                Reportable::Int(val) => writeln!(f, "\t{key}: {val}")?,
                Reportable::Avg(avg) => writeln!(f, "\t{key}: {}", avg.get())?,
            }
        }
        Ok(())
    }
}

impl Report {
    #[cfg(feature = "wandb")]
    pub fn report_wandb(&self, wandb: &mut wandb::run::Run) {
        use wandb::run::Value;

        let mut data = std::collections::HashMap::new();

        for (key, val) in &self.data {
            match val {
                Reportable::Float(v) => {
                    data.insert(key.clone(), Value::Float(*v));
                }
                Reportable::Int(v) => {
                    data.insert(key.clone(), Value::Float(*v as f64));
                }
                Reportable::Avg(avg) => {
                    data.insert(key.clone(), Value::Float(avg.get()));
                }
            }
        }

        wandb.log(data);
    }

    pub fn remove(&mut self, key: &str) {
        self.data.remove(key);
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl Index<String> for Report {
    type Output = Reportable;

    fn index(&self, key: String) -> &Self::Output {
        &self.data[&key]
    }
}

impl Index<&str> for Report {
    type Output = Reportable;

    fn index(&self, key: &str) -> &Self::Output {
        &self.data[key]
    }
}

impl IndexMut<String> for Report {
    fn index_mut(&mut self, key: String) -> &mut Self::Output {
        self.data.entry(key).or_default()
    }
}

impl IndexMut<&str> for Report {
    fn index_mut(&mut self, key: &str) -> &mut Self::Output {
        self.data.entry(key.to_string()).or_default()
    }
}

impl AddAssign<&Report> for Report {
    fn add_assign(&mut self, other: &Report) {
        for (key, val) in other.data.iter() {
            self[key.as_str()] += *val;
        }
    }
}

impl AddAssign<Report> for Report {
    fn add_assign(&mut self, other: Report) {
        for (key, val) in other.data.into_iter() {
            self[key] += val;
        }
    }
}
