use ahash::HashMap;
use std::{
    fmt,
    ops::{AddAssign, Index, IndexMut},
};

use crate::AvgTracker;

#[derive(Debug, Clone, Copy)]
pub enum Reportable {
    Val(f64),
    Avg(AvgTracker),
}

macro_rules! reportable_from_primitive {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Reportable {
                fn from(val: $t) -> Self {
                    Reportable::Val(val as f64)
                }
            }
        )*
    };
}

reportable_from_primitive!(usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128, f32);

impl From<f64> for Reportable {
    fn from(val: f64) -> Self {
        Reportable::Val(val)
    }
}

impl From<AvgTracker> for Reportable {
    fn from(val: AvgTracker) -> Self {
        Reportable::Avg(val)
    }
}

impl Default for Reportable {
    fn default() -> Self {
        Reportable::Val(0.0)
    }
}

impl AddAssign<Reportable> for Reportable {
    fn add_assign(&mut self, other: Reportable) {
        match (self, other) {
            (Reportable::Val(a), Reportable::Val(b)) => *a += b,
            (Reportable::Avg(a), Reportable::Avg(b)) => *a += b,
            (a, b) => *a = b,
        }
    }
}

impl Reportable {
    pub fn as_val(&self) -> f64 {
        match self {
            Reportable::Val(val) => *val,
            Reportable::Avg(_) => unreachable!("Expected Val, got Avg"),
        }
    }

    pub fn as_avg(&self) -> AvgTracker {
        match self {
            Reportable::Avg(avg) => *avg,
            Reportable::Val(_) => unreachable!("Expected Avg, got Val"),
        }
    }

    pub fn as_val_mut(&mut self) -> &mut f64 {
        match self {
            Reportable::Val(val) => val,
            Reportable::Avg(_) => unreachable!("Expected Val, got Avg"),
        }
    }

    pub fn as_avg_mut(&mut self) -> &mut AvgTracker {
        match self {
            Reportable::Avg(avg) => avg,
            Reportable::Val(_) => unreachable!("Expected Avg, got Val"),
        }
    }
}

#[derive(Debug, Default)]
pub struct Report {
    data: HashMap<String, Reportable>,
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            f,
            "-------------------------Iteration report-------------------------"
        )?;
        let mut items: Vec<_> = self.data.iter().collect();
        items.sort_unstable_by(|a, b| a.0.cmp(b.0));

        for (key, val) in items {
            match val {
                Reportable::Val(val) => writeln!(f, "\t{}: {}", key, val)?,
                Reportable::Avg(avg) => writeln!(f, "\t{}: {}", key, avg.get())?,
            }
        }
        Ok(())
    }
}

impl Report {
    pub fn remove(&mut self, key: &str) {
        self.data.remove(key);
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl Index<&str> for Report {
    type Output = Reportable;

    fn index(&self, key: &str) -> &Self::Output {
        &self.data[key]
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
            self[key] += *val;
        }
    }
}
