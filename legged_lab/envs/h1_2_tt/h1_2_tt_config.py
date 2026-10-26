"""H1-2 task config built on the currently verified PACE G1/T1 configs."""
from copy import deepcopy
from isaaclab.utils import configclass

from legged_lab.assets.unitree.h1_2_tt import H1_2_TT_CFG, H1_2_CONTROL_JOINTS
from legged_lab.envs.g1_tt.g1_tt_config import (
    G1TableTennisEnvCfg, G1TableTennisAgentCfg, _make_g1_rewards,
)
from legged_lab.envs.t1_tt.t1_tt_config import T1TT_EvalEnvCfg


def _make_h1_2_rewards():
    reward = _make_g1_rewards()
    reward.undesired_contacts.params["sensor_cfg"].body_names = "(?!(?:.*_ankle_roll_link|h1_2_paddle_link)$).*"
    reward.joint_deviation_left_arms.params["asset_cfg"].joint_names = [
        n for n in H1_2_CONTROL_JOINTS if n.startswith("left_")
        and any(part in n for part in ("shoulder", "elbow", "wrist"))
    ]
    reward.joint_deviation_right_arms.params["asset_cfg"].joint_names = [
        n for n in H1_2_CONTROL_JOINTS if n.startswith("right_")
        and any(part in n for part in ("shoulder", "elbow", "wrist"))
    ]
    # H1-2 has no separate head_link. Keep other existing rewards and record this exception.
    reward.paddel_head_too_near = None
    return reward


@configclass
class H1_2TableTennisEnvCfg(G1TableTennisEnvCfg):
    reward = _make_h1_2_rewards()
    reference_base_height: float = H1_2_TT_CFG.init_state.pos[2]
    reference_paddle_offset_b: tuple = None
    auto_reference_base_height: bool = True
    initial_foot_clearance: float = 0.002

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = H1_2_TT_CFG.copy()
        self.scene.height_scanner.prim_body_name = "torso_link"
        self.robot.terminate_contacts_body_names = ["torso_link"]
        self.robot.feet_body_names = [".*_ankle_roll_link"]
        self.actions.joint_names = list(H1_2_CONTROL_JOINTS)
        self.observations.joint_names = list(H1_2_CONTROL_JOINTS)
        self.actions.preserve_order = True
        self.observations.preserve_order = True
        self.robot.num_actions = 27
        self.robot.num_joints = 27
        # Respect the URDF torque limits used by the asset instead of inherited scaling.
        self.robot.effort_limit_scale = 1.0
        left_arm = [n for n in H1_2_CONTROL_JOINTS if n.startswith("left_") and any(p in n for p in ("shoulder", "elbow", "wrist"))]
        right_arm = [n for n in H1_2_CONTROL_JOINTS if n.startswith("right_") and any(p in n for p in ("shoulder", "elbow", "wrist"))]
        lower = [n for n in H1_2_CONTROL_JOINTS if n not in left_arm + right_arm]
        self.domain_rand.events.reset_locomotion_joints.params["asset_cfg"].joint_names = lower + left_arm
        self.domain_rand.events.reset_manipulation_joints.params["asset_cfg"].joint_names = right_arm
        self.domain_rand.events.physics_material.params["asset_cfg"].body_names = "(?!h1_2_paddle_link$).*"


@configclass
class H1_2TT_EvalEnvCfg(H1_2TableTennisEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        reference = T1TT_EvalEnvCfg()
        self.scene.max_episode_length_s = reference.scene.max_episode_length_s
        self.domain_rand.events.reset_base = deepcopy(reference.domain_rand.events.reset_base)
        self.ball = deepcopy(reference.ball)


@configclass
class H1_2TableTennisAgentCfg(G1TableTennisAgentCfg):
    experiment_name: str = "h1_2_table_tennis"
    resume: bool = False
