use std::path::PathBuf;
use std::sync::{Arc, Once};

use burn::backend::Flex;
use burn::backend::flex::FlexDevice;
use rlbot_rocketsim::rlbot::agents::BotAgent;
use rlbot_rocketsim::rlbot::flat::{
    ControllableInfo, FieldInfo, GamePacket, MatchConfiguration, MatchPhase, PlayerInput,
};
use rlbot_rocketsim::rlbot::util::PacketQueue;
use rlbot_rocketsim::rocketsim::CarControls;
use rlbot_rocketsim::{GameStateEnricher, MatchContext};
use rlgymppo_model::{NormSelection, Policy, PolicyConfig};
use rlgymppo_utils::actions::DefaultAction;
use rlgymppo_utils::rlgym::{Action, Obs};
use rlgymppo_utils::rocketsim::{CarControls as RlgymCarControls, GameMode, init_from_mem};
use rustc_hash::FxHashMap;

use crate::controls::to_rlbot_controls;

use crate::state::to_rlgym_game_state;

static INIT_ROCKETSIM: Once = Once::new();
type Backend = Flex;

const SOCCAR_COLLISION_MESHES: [&[u8]; 16] = [
    include_bytes!("../../collision_meshes/soccar/mesh_0.cmf"),
    include_bytes!("../../collision_meshes/soccar/mesh_1.cmf"),
    include_bytes!("../../collision_meshes/soccar/mesh_2.cmf"),
    include_bytes!("../../collision_meshes/soccar/mesh_3.cmf"),
    include_bytes!("../../collision_meshes/soccar/mesh_4.cmf"),
    include_bytes!("../../collision_meshes/soccar/mesh_5.cmf"),
    include_bytes!("../../collision_meshes/soccar/mesh_6.cmf"),
    include_bytes!("../../collision_meshes/soccar/mesh_7.cmf"),
    include_bytes!("../../collision_meshes/soccar/mesh_8.cmf"),
    include_bytes!("../../collision_meshes/soccar/mesh_9.cmf"),
    include_bytes!("../../collision_meshes/soccar/mesh_10.cmf"),
    include_bytes!("../../collision_meshes/soccar/mesh_11.cmf"),
    include_bytes!("../../collision_meshes/soccar/mesh_12.cmf"),
    include_bytes!("../../collision_meshes/soccar/mesh_13.cmf"),
    include_bytes!("../../collision_meshes/soccar/mesh_14.cmf"),
    include_bytes!("../../collision_meshes/soccar/mesh_15.cmf"),
];

fn soccar_collision_meshes() -> FxHashMap<GameMode, Vec<Vec<u8>>> {
    FxHashMap::from_iter([(
        GameMode::Soccar,
        SOCCAR_COLLISION_MESHES
            .iter()
            .map(|mesh| mesh.to_vec())
            .collect(),
    )])
}

/// Converts a discrete policy action into RocketSim controls for RLBot.
pub trait RlbotAction {
    fn get_action(&self, action_index: usize) -> RlgymCarControls;
}

impl<const MAX_PLAYERS: usize, const TICK_SKIP: u8> RlbotAction
    for DefaultAction<MAX_PLAYERS, TICK_SKIP>
{
    fn get_action(&self, action_index: usize) -> RlgymCarControls {
        self.get_action(action_index)
    }
}

/// RLBot agent using configurable RLGym observation and action implementations.
pub struct PpoBot<OBS, ACT, SI> {
    player_index: usize,
    enricher: GameStateEnricher,
    observation: OBS,
    action_parser: ACT,
    shared_info: SI,
    policy: Policy<Backend>,
    device: FlexDevice,
    held_controls: CarControls,
    next_decision_frame: Option<u32>,
}

impl<OBS, ACT, SI> BotAgent for PpoBot<OBS, ACT, SI>
where
    OBS: Obs<SI> + Default + Send + 'static,
    ACT: Action<SI, Input = usize> + RlbotAction + Default + Send + 'static,
    SI: From<u64> + Send + 'static,
{
    fn new(
        _team: u32,
        controllable_info: ControllableInfo,
        match_config: Arc<MatchConfiguration>,
        field_info: Arc<FieldInfo>,
        _packet_queue: &mut PacketQueue,
    ) -> Self {
        INIT_ROCKETSIM.call_once(|| {
            init_from_mem(soccar_collision_meshes(), true)
                .expect("initialize embedded RocketSim Soccar collision meshes");
        });

        let context = MatchContext::new(&match_config, &field_info)
            .expect("create RocketSim context from RLBot match");
        let device = Default::default();
        let checkpoint = checkpoint_path();
        let observation = OBS::default();
        let action_parser = ACT::default();
        let shared_info = SI::from(controllable_info.identifier as u64);
        let policy = Policy::<Backend>::load(
            &checkpoint,
            &PolicyConfig {
                input_size: observation.get_obs_space(&shared_info),
                action_size: action_parser.get_action_space(&shared_info),
                actor_layer_sizes: vec![256; 2],
                shared_head_layer_sizes: vec![256; 3],
                norm: NormSelection::RmsNorm,
            },
            &device,
        )
        .unwrap_or_else(|error| {
            panic!(
                "load policy checkpoint from {}: {error}",
                checkpoint.display()
            )
        });

        Self {
            player_index: controllable_info.index as usize,
            enricher: GameStateEnricher::from_match_context(context),
            observation,
            action_parser,
            shared_info,
            policy,
            device,
            held_controls: CarControls::default(),
            next_decision_frame: None,
        }
    }

    fn tick(&mut self, packet: &GamePacket, packet_queue: &mut PacketQueue) {
        let frame = packet.match_info.frame_num;
        let enriched = self.enricher.update(packet).is_ok();
        if enriched
            && matches!(
                packet.match_info.match_phase,
                MatchPhase::Kickoff | MatchPhase::Active
            )
            && packet.players.get(self.player_index).is_some()
            && self
                .next_decision_frame
                .is_none_or(|next_frame| frame >= next_frame)
        {
            self.infer(frame);
        }

        packet_queue.push(PlayerInput {
            player_index: self.player_index as u32,
            controller_state: to_rlbot_controls(self.held_controls),
        });
    }
}

impl<OBS, ACT, SI> PpoBot<OBS, ACT, SI>
where
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize> + RlbotAction,
{
    fn infer(&mut self, frame: u32) {
        let state = to_rlgym_game_state(self.enricher.arena_state());
        let observations = self.observation.build_obs(&state, &mut self.shared_info);
        let masks = self
            .action_parser
            .get_action_masks(&state, &mut self.shared_info);
        let action = self.policy.react_deterministic_indexed(
            &observations,
            &masks,
            &[self.player_index],
            &self.device,
        )[0];

        self.held_controls = self.action_parser.get_action(action);
        self.next_decision_frame = Some(frame + ACT::get_tick_skip() as u32);
    }
}

fn checkpoint_path() -> PathBuf {
    if let Some(path) = std::env::var_os("RLGYMPPO_CHECKPOINT") {
        return PathBuf::from(path);
    }

    let checkpoints =
        find_asset_directory("checkpoints").unwrap_or_else(|| PathBuf::from("checkpoints"));
    checkpoints
        .read_dir()
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| {
            entry
                .file_name()
                .to_str()
                .and_then(|name| name.parse::<u64>().ok())
                .map(|timestamp| (timestamp, entry.path()))
        })
        .max_by_key(|(timestamp, _)| *timestamp)
        .map(|(_, path)| path)
        .unwrap_or(checkpoints)
}

fn find_asset_directory(name: &str) -> Option<PathBuf> {
    let current = std::env::current_dir().ok()?;
    current
        .ancestors()
        .map(|ancestor| ancestor.join(name))
        .find(|candidate| candidate.is_dir())
}

#[cfg(test)]
mod tests {
    use rlgymppo_utils::rocketsim::{Arena, GameMode};

    use super::*;

    #[test]
    fn embedded_soccar_meshes_initialize() {
        init_from_mem(soccar_collision_meshes(), true).unwrap();
        let arena = Arena::new(GameMode::Soccar);

        assert_eq!(arena.game_mode(), GameMode::Soccar);
    }
}
