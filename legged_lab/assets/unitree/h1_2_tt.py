"""H1-2 PACE asset configuration; gains and standing height are provisional."""
from pathlib import Path
import json

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_HERE = Path(__file__).resolve().parent
_SPEC = json.loads((_HERE / "H1_2_BASE/joint_spec.json").read_text())
H1_2_JOINT_SPECS = _SPEC["joint_specs"]
H1_2_CONTROL_JOINTS = [
    f"{side}_{name}_joint"
    for side in ("left", "right")
    for name in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw",
    )
] + ["torso_joint"] + [
    f"{side}_{name}_joint"
    for side in ("left", "right")
    for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
]
assert len(H1_2_CONTROL_JOINTS) == 27
assert set(H1_2_CONTROL_JOINTS) == set(H1_2_JOINT_SPECS)

H1_2_PADDLE_PARENT_BODY_NAME = "right_wrist_yaw_link"
H1_2_PADDLE_BODY_NAME = "h1_2_paddle_link"
_PADDLE_REPORT = json.loads((_HERE / "H1_2_TT/paddle_asset_report.json").read_text())
H1_2_PADDLE_PARENT_OFFSET = tuple(_PADDLE_REPORT["parent_offset"])
H1_2_PADDLE_CENTER_OFFSET = (0.0, 0.0, 0.0)
H1_2_PADDLE_NORMAL = tuple(_PADDLE_REPORT["local_normal"])
H1_2_PADDLE_RADIUS = float(_PADDLE_REPORT["head_radius"])
H1_2_PADDLE_HALF_THICKNESS = float(_PADDLE_REPORT["head_thickness"]) / 2.0
H1_2_INITIAL_BASE_HEIGHT = 1.0  # provisional; ground-supported calibration pending

_DEFAULT = {name: 0.0 for name in H1_2_CONTROL_JOINTS}
for _side in ("left", "right"):
    _DEFAULT.update({
        f"{_side}_hip_pitch_joint": -0.28,
        f"{_side}_knee_joint": 0.56,
        f"{_side}_ankle_pitch_joint": -0.28,
        f"{_side}_shoulder_pitch_joint": 0.25,
        f"{_side}_shoulder_roll_joint": 0.10 if _side == "left" else -0.10,
        f"{_side}_elbow_joint": 0.70,
    })
for _name, _value in _DEFAULT.items():
    assert H1_2_JOINT_SPECS[_name]["lower"] <= _value <= H1_2_JOINT_SPECS[_name]["upper"]

# Simulation gains, not manufacturer-certified controller settings.
_GROUPS = {
    "legs_and_torso": (
        [n for n in H1_2_CONTROL_JOINTS if "_hip_" in n or "_knee_" in n or n == "torso_joint"],
        200.0, 5.0, 0.01,
    ),
    "ankles": ([n for n in H1_2_CONTROL_JOINTS if "_ankle_" in n], 60.0, 3.0, 0.01),
    "shoulders": ([n for n in H1_2_CONTROL_JOINTS if "_shoulder_" in n], 60.0, 3.0, 0.01),
    "elbows": ([n for n in H1_2_CONTROL_JOINTS if n.endswith("_elbow_joint")], 40.0, 2.0, 0.01),
    "wrists": ([n for n in H1_2_CONTROL_JOINTS if "_wrist_" in n], 20.0, 1.0, 0.001),
}
_COVERAGE = [n for names, _, _, _ in _GROUPS.values() for n in names]
assert len(_COVERAGE) == 27 and set(_COVERAGE) == set(H1_2_CONTROL_JOINTS)

H1_2_TT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_HERE / "H1_2_TT/H1_2_TT.usda"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-1.6, 0.0, H1_2_INITIAL_BASE_HEIGHT),
        joint_pos=_DEFAULT,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        key: ImplicitActuatorCfg(
            joint_names_expr=names,
            effort_limit_sim={n: H1_2_JOINT_SPECS[n]["effort"] for n in names},
            velocity_limit_sim={n: H1_2_JOINT_SPECS[n]["velocity"] for n in names},
            stiffness=stiffness,
            damping=damping,
            armature=armature,
        )
        for key, (names, stiffness, damping, armature) in _GROUPS.items()
    },
)
