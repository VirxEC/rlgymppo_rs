use rand::SeedableRng;
use rand::rngs::SmallRng;
use rlbot_rocketsim::rlbot::RLBotConnection;
use rlbot_rocketsim::rlbot::agents::run_bot_agents;
use rlbot_rocketsim::rlbot::util::AgentEnvironment;
use rlgymppo_rlbot::PpoBot;
use rlgymppo_utils::actions::DefaultAction;
use rlgymppo_utils::obs::DefaultObs;
use rlgymppo_utils::shared_info::SharedInfoRng;

struct SharedInfo {
    rng: SmallRng,
}

impl From<u64> for SharedInfo {
    fn from(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl SharedInfoRng for SharedInfo {
    type Rng = SmallRng;

    fn rng(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

// Configure the shared info, observation builder, and discrete action parser used by this bot.
type Bot = PpoBot<DefaultObs<3>, DefaultAction<6, 8>, SharedInfo>;

fn main() {
    let AgentEnvironment {
        server_addr,
        agent_id,
    } = AgentEnvironment::from_env();
    let agent_id = agent_id.unwrap_or_else(|| "rlgymppo-rs/ppo-bot".into());
    let connection = RLBotConnection::new(&server_addr).expect("connect to RLBot");

    run_bot_agents::<Bot>(agent_id, false, false, connection).expect("run rlgymppo RLBot agent");
}
