use std::ops::AddAssign;

#[derive(Debug, Clone, Copy, Default)]
pub struct AvgTracker {
    total: f64,
    count: u64,
}

impl AvgTracker {
    pub fn new(total: f64, count: u64) -> Self {
        Self { total, count }
    }

    pub fn get(&self) -> f64 {
        if self.count > 0 {
            self.total / self.count as f64
        } else {
            f64::NAN
        }
    }

    pub fn reset(&mut self) {
        self.total = 0.0;
        self.count = 0;
    }
}

impl AddAssign<f64> for AvgTracker {
    fn add_assign(&mut self, val: f64) {
        if !val.is_nan() {
            self.total += val;
            self.count += 1;
        }
    }
}

impl AddAssign<AvgTracker> for AvgTracker {
    fn add_assign(&mut self, other: AvgTracker) {
        if !other.total.is_nan() {
            self.total += other.total;
            self.count += other.count;
        }
    }
}
