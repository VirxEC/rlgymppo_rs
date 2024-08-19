use std::ops::{AddAssign, Index, IndexMut};

use ahash::HashMap;

#[derive(Default)]
pub struct Report {
    data: HashMap<String, f64>,
}

impl Report {
    pub fn acum_avg(&mut self, key: &str, val: f64) {
        self[&(key.to_string() + "_avg_total")] += val;
        self[&(key.to_string() + "_avg_count")] += 1.0;
    }

    pub fn get_avg(&self, key: &str) -> f64 {
        let total = self[&(key.to_string() + "_avg_total")];
        let count = self[&(key.to_string() + "_avg_count")];

        if count > 0.0 {
            total / count
        } else {
            0.0
        }
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl Index<&str> for Report {
    type Output = f64;

    fn index(&self, key: &str) -> &Self::Output {
        &self.data[key]
    }
}

impl IndexMut<&str> for Report {
    fn index_mut(&mut self, key: &str) -> &mut Self::Output {
        self.data.entry(key.to_string()).or_insert(0.0)
    }
}

impl AddAssign<&Report> for Report {
    fn add_assign(&mut self, other: &Report) {
        for (key, val) in other.data.iter() {
            self[key] += val;
        }
    }
}
