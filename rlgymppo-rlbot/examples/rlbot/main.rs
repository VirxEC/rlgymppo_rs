use rand::SeedableRng;
use rand::rngs::SmallRng;
use rlbot_rocketsim::rlbot::RLBotConnection;
use rlbot_rocketsim::rlbot::agents::run_bot_agents;
use rlbot_rocketsim::rlbot::util::AgentEnvironment;
use rlgymppo_model::{NormSelection, PolicyConfig};
use rlgymppo_rlbot::{ConfigurablePpoBot, PpoBotConfig};
use rlgymppo_utils::actions::DefaultAction;
use rlgymppo_utils::obs::DefaultObs;
use rlgymppo_utils::rocketsim::{GameMode, init_from_mem};
use rlgymppo_utils::shared_info::SharedInfoRng;
use rustc_hash::FxHashMap;

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

/// Default example architecture:
/// shared head `[256, 256, 256]`, actor `[256, 256]`, RMSNorm.
struct ExampleConfig;

impl PpoBotConfig for ExampleConfig {
    fn policy_config(input_size: usize, action_size: usize) -> PolicyConfig {
        PolicyConfig {
            input_size,
            action_size,
            actor_layer_sizes: vec![256; 2],
            shared_head_layer_sizes: vec![256; 3],
            norm: NormSelection::RmsNorm,
        }
    }
}

// Configure the shared info, observation builder, discrete action parser, and policy architecture.
type Bot = ConfigurablePpoBot<DefaultObs<3>, DefaultAction<6, 8, 1>, SharedInfo, ExampleConfig>;

const SOCCAR_COLLISION_MESHES: [&[u8]; 16] = [
    include_bytes!("../../../collision_meshes/soccar/mesh_0.cmf"),
    include_bytes!("../../../collision_meshes/soccar/mesh_1.cmf"),
    include_bytes!("../../../collision_meshes/soccar/mesh_2.cmf"),
    include_bytes!("../../../collision_meshes/soccar/mesh_3.cmf"),
    include_bytes!("../../../collision_meshes/soccar/mesh_4.cmf"),
    include_bytes!("../../../collision_meshes/soccar/mesh_5.cmf"),
    include_bytes!("../../../collision_meshes/soccar/mesh_6.cmf"),
    include_bytes!("../../../collision_meshes/soccar/mesh_7.cmf"),
    include_bytes!("../../../collision_meshes/soccar/mesh_8.cmf"),
    include_bytes!("../../../collision_meshes/soccar/mesh_9.cmf"),
    include_bytes!("../../../collision_meshes/soccar/mesh_10.cmf"),
    include_bytes!("../../../collision_meshes/soccar/mesh_11.cmf"),
    include_bytes!("../../../collision_meshes/soccar/mesh_12.cmf"),
    include_bytes!("../../../collision_meshes/soccar/mesh_13.cmf"),
    include_bytes!("../../../collision_meshes/soccar/mesh_14.cmf"),
    include_bytes!("../../../collision_meshes/soccar/mesh_15.cmf"),
];

fn main() {
    let collision_meshes = FxHashMap::from_iter([(
        GameMode::Soccar,
        SOCCAR_COLLISION_MESHES
            .iter()
            .map(|mesh| mesh.to_vec())
            .collect(),
    )]);
    init_from_mem(collision_meshes, true)
        .expect("initialize embedded RocketSim Soccar collision meshes");

    let AgentEnvironment {
        server_addr,
        agent_id,
    } = AgentEnvironment::from_env();
    let agent_id = agent_id.unwrap_or_else(|| "rlgymppo-rs/ppo-bot".into());
    let connection = RLBotConnection::new(&server_addr).expect("connect to RLBot");

    run_bot_agents::<Bot>(agent_id, false, false, connection).expect("run rlgymppo RLBot agent");
}
