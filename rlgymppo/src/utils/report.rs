use std::{
    fmt,
    ops::{AddAssign, Index, IndexMut},
};

use rustc_hash::FxHashMap;

use crate::utils::AvgTracker;

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
                    Self::Int(val as i64)
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
        Self::Float(val)
    }
}

impl From<f32> for Reportable {
    fn from(val: f32) -> Self {
        Self::Float(val as f64)
    }
}

impl From<AvgTracker> for Reportable {
    fn from(val: AvgTracker) -> Self {
        Self::Avg(val)
    }
}

impl Default for Reportable {
    fn default() -> Self {
        Self::Float(0.0)
    }
}

impl AddAssign<Reportable> for Reportable {
    fn add_assign(&mut self, other: Reportable) {
        match (self, other) {
            (Self::Float(a), Self::Float(b)) => *a += b,
            (Self::Int(a), Self::Int(b)) => *a += b,
            (Self::Avg(a), Self::Avg(b)) => *a += b,
            (a, b) => *a = b,
        }
    }
}

// impl AddAssign<T> for Reportable where T is the different values of Reportable
impl AddAssign<f64> for Reportable {
    fn add_assign(&mut self, other: f64) {
        match self {
            Self::Float(a) => *a += other,
            Self::Avg(a) => *a += AvgTracker::from(other),
            _ => panic!("Expected Float, got {self:?}"),
        }
    }
}

impl AddAssign<f32> for Reportable {
    fn add_assign(&mut self, other: f32) {
        match self {
            Self::Float(a) => *a += other as f64,
            Self::Avg(a) => *a += AvgTracker::from(other),
            _ => panic!("Expected Float, got {self:?}"),
        }
    }
}

impl AddAssign<i64> for Reportable {
    fn add_assign(&mut self, other: i64) {
        match self {
            Self::Int(a) => *a += other,
            Self::Avg(a) => *a += AvgTracker::from(other as f64),
            _ => panic!("Expected Int, got {self:?}"),
        }
    }
}

impl AddAssign<AvgTracker> for Reportable {
    fn add_assign(&mut self, mut other: AvgTracker) {
        match self {
            Self::Float(a) => {
                other += *a;
                *self = other.into();
            }
            Self::Int(a) => {
                other += *a as f64;
                *self = other.into();
            }
            Self::Avg(a) => *a += other,
        }
    }
}

impl Reportable {
    pub fn as_float(&self) -> f64 {
        match self {
            Self::Float(val) => *val,
            _ => unreachable!("Expected Val, got Avg"),
        }
    }

    pub fn as_int(&self) -> i64 {
        match self {
            Self::Int(val) => *val,
            _ => unreachable!("Expected Int, got Float"),
        }
    }

    pub fn as_avg(&self) -> AvgTracker {
        match self {
            Self::Avg(avg) => *avg,
            _ => unreachable!("Expected Avg, got Val"),
        }
    }

    pub fn as_float_mut(&mut self) -> &mut f64 {
        match self {
            Self::Float(val) => val,
            _ => unreachable!("Expected Val, got Avg"),
        }
    }

    pub fn as_int_mut(&mut self) -> &mut i64 {
        match self {
            Self::Int(val) => val,
            _ => unreachable!("Expected Int, got Float"),
        }
    }

    pub fn as_avg_mut(&mut self) -> &mut AvgTracker {
        match self {
            Self::Avg(avg) => avg,
            _ => unreachable!("Expected Avg, got Val"),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    data: FxHashMap<String, Reportable>,
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
    /// Add a floating-point value under an average-tracking key (e.g. `"Rewards/boost"`).
    /// Subsequent calls with the same key will be averaged together.
    pub fn add_avg(&mut self, key: &str, value: f64) {
        self[key] += AvgTracker::new(value, 1);
    }

    /// Convert the report to a flat `{name → value}` map suitable for
    /// logging (e.g. via [`MetricSender`](rlgymppo_wandb::MetricSender)).
    /// Average-tracker values are resolved to their current mean.
    pub fn to_flat_map(&self) -> std::collections::HashMap<String, f64> {
        let mut map = std::collections::HashMap::new();
        for (key, val) in &self.data {
            match val {
                Reportable::Float(v) => {
                    map.insert(key.clone(), *v);
                }
                Reportable::Int(v) => {
                    map.insert(key.clone(), *v as f64);
                }
                Reportable::Avg(avg) => {
                    map.insert(key.clone(), avg.get());
                }
            }
        }
        map
    }

    pub fn remove(&mut self, key: &str) {
        self.data.remove(key);
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Remove all entries whose key starts with the given prefix.
    pub fn remove_keys_with_prefix(&mut self, prefix: &str) {
        self.data.retain(|key, _| !key.starts_with(prefix));
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
