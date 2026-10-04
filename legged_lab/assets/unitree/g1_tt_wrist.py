"""G1 paddle variant with one active right-palm hinge."""
from pathlib import Path
from isaaclab.actuators import ImplicitActuatorCfg
from legged_lab.assets.unitree.g1_tt import G1_TT_CFG

G1_WRIST_JOINT_NAME = "right_palm_joint"
G1_TT_WRIST_CFG = G1_TT_CFG.copy()
G1_TT_WRIST_CFG.spawn.usd_path = str(Path(__file__).resolve().parent / "G1_TT_WRIST" / "G1_TT_WRIST.usda")
G1_TT_WRIST_CFG.init_state.joint_pos[G1_WRIST_JOINT_NAME] = 0.0
G1_TT_WRIST_CFG.actuators["right_wrist"] = ImplicitActuatorCfg(
    joint_names_expr=[G1_WRIST_JOINT_NAME],
    effort_limit_sim=8.0,
    velocity_limit_sim=6.0,
    stiffness=20.0,
    damping=1.0,
    armature=0.001,
)
