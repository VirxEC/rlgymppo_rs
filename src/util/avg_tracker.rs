use std::ops::AddAssign;

#[derive(Clone, Copy, Default)]
pub struct AvgTracker {
    total: f32,
    count: u64,
}

impl AvgTracker {
    pub fn new(total: f32, count: u64) -> Self {
        Self { total, count }
    }

    pub fn get(&self) -> f32 {
        if self.count > 0 {
            self.total / self.count as f32
        } else {
            f32::NAN
        }
    }

    pub fn reset(&mut self) {
        self.total = 0.0;
        self.count = 0;
    }
}

impl AddAssign<f32> for AvgTracker {
    fn add_assign(&mut self, val: f32) {
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
