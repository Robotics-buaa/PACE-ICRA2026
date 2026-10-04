"""24-action G1 right-wrist task; old G1 tasks stay available."""
from isaaclab.utils import configclass
from legged_lab.assets.unitree.g1_tt_wrist import G1_TT_WRIST_CFG, G1_WRIST_JOINT_NAME
from legged_lab.envs.g1_tt.g1_tt_config import (
    G1_CONTROL_JOINTS, G1_HAND_JOINTS, G1TableTennisEnvCfg,
    G1TT_EvalEnvCfg, G1TableTennisAgentCfg,
)

G1_WRIST_CONTROL_JOINTS = list(G1_CONTROL_JOINTS) + [G1_WRIST_JOINT_NAME]


def _configure_wrist(cfg):
    cfg.scene.robot = G1_TT_WRIST_CFG.copy()
    cfg.actions.joint_names = list(G1_WRIST_CONTROL_JOINTS)
    cfg.observations.joint_names = list(G1_WRIST_CONTROL_JOINTS)
    cfg.actions.preserve_order = True
    cfg.observations.preserve_order = True
    cfg.robot.num_actions = len(G1_WRIST_CONTROL_JOINTS)
    cfg.robot.num_joints = len(G1_WRIST_CONTROL_JOINTS) + len(G1_HAND_JOINTS)
    # Wrist reset is explicit in G1WristTTEnv. Existing arm reset groups keep
    # their current ranges and the new wrist starts at its neutral angle.


@configclass
class G1WristTableTennisEnvCfg(G1TableTennisEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_wrist(self)


@configclass
class G1WristTT_EvalEnvCfg(G1TT_EvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_wrist(self)


@configclass
class G1WristTableTennisAgentCfg(G1TableTennisAgentCfg):
    experiment_name: str = "g1_table_tennis_wrist"
    resume: bool = False
