mod combined;
mod common;
mod zero_sum;

pub use combined::CombinedRewards;
pub use common::{
    AirReward, BallTouchReward, BumpReward, BumpedPenalty, DemoReward, DemoedPenalty,
    FaceBallReward, GoalReward, PickupBoostReward, SaveBoostReward, StrongTouchReward,
    TouchAccelReward, VelocityBallToGoalReward, VelocityReward, VelocityToBallReward,
    WavedashReward,
};
pub use zero_sum::ZeroSumReward;
