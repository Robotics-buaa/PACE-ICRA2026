"""Convert right_palm_joint in a separate local G1 paddle variant.

Run --stage asset first; run --stage task after its physical check passes.
No training, installation, or checkpoint loading is performed.
"""
import argparse
import ast
import hashlib
import json
import math
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

JOINT = "right_palm_joint"
ASSET_CONFIG = '''"""G1 paddle variant with one active right-palm hinge."""
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
'''
TASK_CONFIG = '''"""24-action G1 right-wrist task; old G1 tasks stay available."""
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
'''
TASK_ENV = '''"""Use the G1 task interfaces with one controlled right-wrist joint."""
from legged_lab.envs.g1_tt.g1_tt_env import G1TTEnv
from legged_lab.assets.unitree.g1_tt_wrist import G1_WRIST_JOINT_NAME


class G1WristTTEnv(G1TTEnv):
    def init_buffers(self):
        super().init_buffers()
        ids, names = self.robot.find_joints(G1_WRIST_JOINT_NAME)
        if names != [G1_WRIST_JOINT_NAME] or len(ids) != 1:
            raise RuntimeError("Expected one active right_palm_joint")
        self.wrist_joint_ids = ids
        if ids[0] not in self.action_joint_ids or ids[0] in self.held_joint_ids:
            raise RuntimeError("Wrist must be policy-controlled, not held with fingers")

    def reset(self, env_ids):
        super().reset(env_ids)
        if len(env_ids) == 0:
            return
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids][:, self.wrist_joint_ids],
            self.robot.data.default_joint_vel[env_ids][:, self.wrist_joint_ids],
            joint_ids=self.wrist_joint_ids, env_ids=env_ids,
        )
        self.robot.set_joint_position_target(
            self.robot.data.default_joint_pos[env_ids][:, self.wrist_joint_ids],
            joint_ids=self.wrist_joint_ids, env_ids=env_ids,
        )
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.compute_paddle_touch()
'''
START = "# BEGIN PACE G1 WRIST TASKS"
END = "# END PACE G1 WRIST TASKS"
REGISTRATION = '''# BEGIN PACE G1 WRIST TASKS
from legged_lab.envs.g1_tt_wrist.g1_tt_wrist_env import G1WristTTEnv
from legged_lab.envs.g1_tt_wrist.g1_tt_wrist_config import (
    G1WristTableTennisEnvCfg, G1WristTT_EvalEnvCfg, G1WristTableTennisAgentCfg,
)
task_registry.register("g1_tt_wrist", G1WristTTEnv, G1WristTableTennisEnvCfg(), G1WristTableTennisAgentCfg())
task_registry.register("g1_tt_wrist_eval", G1WristTTEnv, G1WristTT_EvalEnvCfg(), G1WristTableTennisAgentCfg())
# END PACE G1 WRIST TASKS
'''


def write_new(path, source):
    """Do not overwrite custom edits to any variant source on a repeat run."""
    if path.exists():
        if path.read_text(encoding="utf-8") != source:
            raise RuntimeError("Existing different file preserved: " + str(path))
        return
    if path.suffix == ".py":
        compile(source, str(path), "exec")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def protected_snapshot(project):
    names = (
        "legged_lab/assets/unitree/g1_tt.py", "legged_lab/assets/unitree/G1_TT/G1_TT.usda",
        "legged_lab/envs/g1_tt/g1_tt_config.py", "legged_lab/envs/g1_tt/g1_tt_env.py",
        "legged_lab/envs/t1_tt/t1_tt_config.py", "legged_lab/envs/base/tt_env.py",
        "legged_lab/mdp/rewards.py", "legged_lab/physics/aerodynamics.py",
    )
    return {name: hashlib.sha256((project / name).read_bytes()).hexdigest()
            for name in names if (project / name).is_file()}


def build_asset(project):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from legged_lab.assets.unitree.g1_tt import G1_TT_CFG, G1_PADDLE_NORMAL
    directory = project / "legged_lab/assets/unitree/G1_TT_WRIST"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "G1_TT_WRIST.usda"
    config = project / "legged_lab/assets/unitree/g1_tt_wrist.py"
    if config.exists() and config.read_text(encoding="utf-8") != ASSET_CONFIG:
        raise RuntimeError("Existing wrist asset configuration preserved: " + str(config))
    source = Path(G1_TT_CFG.spawn.usd_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/G1").GetPrim()
    root.GetReferences().AddReference(str(source))
    root.SetInstanceable(False)
    stage.SetDefaultPrim(root)
    found = [prim for prim in stage.Traverse() if prim.GetName() == JOINT]
    if len(found) != 1:
        raise RuntimeError("Expected one right_palm_joint; found %d" % len(found))
    prim = found[0]
    if not prim.IsA(UsdPhysics.FixedJoint):
        raise RuntimeError("Original right_palm_joint is not PhysicsFixedJoint: " + prim.GetTypeName())
    fixed = UsdPhysics.FixedJoint(prim)
    body0, body1 = fixed.GetBody0Rel().GetTargets(), fixed.GetBody1Rel().GetTargets()
    if len(body0) != 1 or len(body1) != 1:
        raise RuntimeError("Wrist must connect exactly two bodies")
    pos0 = fixed.GetLocalPos0Attr().Get() or Gf.Vec3f(0)
    pos1 = fixed.GetLocalPos1Attr().Get() or Gf.Vec3f(0)
    rot0 = fixed.GetLocalRot0Attr().Get() or Gf.Quatf(1)
    rot1 = fixed.GetLocalRot1Attr().Get() or Gf.Quatf(1)
    original = {
        "joint_path": str(prim.GetPath()), "type": prim.GetTypeName(),
        "body0": str(body0[0]), "body1": str(body1[0]),
        "local_pos0": list(pos0), "local_pos1": list(pos1),
        "local_rot0_wxyz": [float(rot0.GetReal()), *list(rot0.GetImaginary())],
        "local_rot1_wxyz": [float(rot1.GetReal()), *list(rot1.GetImaginary())],
    }
    # Put the palm on body1 so positive hinge motion is expressed in its frame.
    swapped = False
    if stage.GetPrimAtPath(body1[0]).GetName() != "right_palm_link":
        if stage.GetPrimAtPath(body0[0]).GetName() != "right_palm_link":
            raise RuntimeError("right_palm_joint does not connect right_palm_link")
        body0, body1 = body1, body0
        pos0, pos1 = pos1, pos0
        rot0, rot1 = rot1, rot0
        swapped = True
    for path in (body0[0], body1[0]):
        if not stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("Joint endpoint is not a rigid body: " + str(path))
    # Re-express the same zero-angle joint frames with body1 rotation identity:
    # Q_parent * (rot0 * inverse(rot1)) = Q_palm. Anchors stay unchanged.
    zero_rotation = (rot0 * rot1.GetInverse()).GetNormalized()
    joint = UsdPhysics.RevoluteJoint.Define(stage, prim.GetPath())
    joint.CreateBody0Rel().SetTargets(body0)
    joint.CreateBody1Rel().SetTargets(body1)
    joint.CreateLocalPos0Attr(pos0)
    joint.CreateLocalPos1Attr(pos1)
    joint.CreateLocalRot0Attr(zero_rotation)
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    # Palm-local Z is perpendicular to the existing palm-local Y paddle normal.
    # It tilts the face rather than spinning it about its own normal.
    if abs(float(G1_PADDLE_NORMAL[2])) > 1e-6:
        raise RuntimeError("Expected current paddle normal (0,1,0); inspect custom asset")
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-90.0)
    joint.CreateUpperLimitAttr(90.0)
    joint.CreateJointEnabledAttr(True)
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateExcludeFromArticulationAttr(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    # USD angular spring/damper use degrees; Isaac Lab actuator config uses radians.
    drive.CreateStiffnessAttr(20.0 * math.pi / 180.0)
    drive.CreateDampingAttr(1.0 * math.pi / 180.0)
    drive.CreateMaxForceAttr(8.0)
    if output.exists():
        backup = output.with_name(output.name + "." + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".bak")
        shutil.copy2(output, backup)
        print("Backup:", backup, flush=True)
    assert stage.GetRootLayer().Export(str(output))
    write_new(config, ASSET_CONFIG)
    report = {
        "source_usd": str(source), "variant_usd": str(output), "original": original,
        "parent_body": str(body0[0]), "palm_body": str(body1[0]), "endpoints_swapped": swapped,
        "zero_parent_to_palm_quat_wxyz": [float(zero_rotation.GetReal()), *list(zero_rotation.GetImaginary())],
        "axis": "Z", "axis_frame": "right_palm_link at zero joint angle",
        "limits_deg": [-90, 90], "actuator": {"stiffness": 20, "damping": 1,
        "effort_limit_sim": 8, "velocity_limit_sim": 6, "armature": 0.001},
        "note": "A simulated extra wrist DOF; parameter choices are test values, not hardware specifications.",
    }
    (directory / "wrist_joint_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[PASS] Original joint:", json.dumps(original), flush=True)
    print("[PASS] Variant right_palm_joint: RevoluteJoint, palm-local Z, +/-90 deg", flush=True)
    print("[PASS] Original anchors/zero pose preserved; paddle joint remains fixed", flush=True)
    return report


def check_asset(project, report, gui, app):
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul, quat_rotate_inverse
    from legged_lab.assets.unitree.g1_tt import (
        G1_PADDLE_BODY_NAME, G1_PADDLE_NORMAL, G1_PADDLE_PARENT_OFFSET,
    )
    from legged_lab.assets.unitree.g1_tt_wrist import G1_TT_WRIST_CFG
    from legged_lab.envs.g1_tt.g1_tt_config import G1_CONTROL_JOINTS, G1_HAND_JOINTS
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.002, device="cuda:0"))
    # Only this isolated mechanics test uses a fixed root and disabled gravity.
    cfg = G1_TT_WRIST_CFG.copy()
    cfg.spawn.articulation_props.fix_root_link = True
    cfg.spawn.rigid_props.disable_gravity = True
    robot = Articulation(cfg.replace(prim_path="/World/G1Wrist"))
    sim.reset()
    assert robot.num_joints == 38
    assert set(robot.joint_names) == set(G1_CONTROL_JOINTS + G1_HAND_JOINTS + [JOINT])
    wrist_ids, names = robot.find_joints(JOINT)
    assert names == [JOINT] and len(wrist_ids) == 1
    wrist = wrist_ids[0]
    group = robot.actuators["right_wrist"]
    assert group.joint_names == [JOINT]
    limits = robot.data.joint_pos_limits[0, wrist]
    assert torch.allclose(limits, limits.new_tensor([-math.pi/2, math.pi/2]), atol=1e-4)
    palm_id = robot.find_bodies("right_palm_link")[0][0]
    parent_name = report["parent_body"].rsplit("/", 1)[-1]
    parent_id = robot.find_bodies(parent_name)[0][0]
    paddle_id = robot.find_bodies(G1_PADDLE_BODY_NAME)[0][0]
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    targets = robot.data.default_joint_pos.clone()
    local_normal = targets.new_tensor(G1_PADDLE_NORMAL).unsqueeze(0)
    expected_offset = targets.new_tensor(G1_PADDLE_PARENT_OFFSET).unsqueeze(0)
    max_attachment_error = 0.0
    samples = []

    def advance(angle):
        nonlocal max_attachment_error
        targets[:, wrist] = angle
        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        sim.step(render=gui)
        robot.update(sim.get_physics_dt())
        pq = robot.data.body_quat_w[:, palm_id]
        relative = quat_rotate_inverse(pq, robot.data.body_pos_w[:, paddle_id] - robot.data.body_pos_w[:, palm_id])
        max_attachment_error = max(max_attachment_error, float((relative - expected_offset).norm()))
        assert bool(torch.isfinite(robot.data.joint_pos).all())
        assert float((pq * robot.data.body_quat_w[:, paddle_id]).sum(-1).abs().min()) > 0.995

    for angle in (0.0, -0.8, 0.8, 0.0):
        for _ in range(400):
            advance(angle)
        actual = float(robot.data.joint_pos[0, wrist])
        assert abs(actual - angle) < 0.05, ("Wrist tracking error", angle, actual)
        normal = quat_apply(robot.data.body_quat_w[:, paddle_id], local_normal)
        normal_parent = quat_rotate_inverse(robot.data.body_quat_w[:, parent_id], normal)
        samples.append(normal_parent[0].detach().clone())
        print("[WRIST] target=%.3frad actual=%.3frad normal_world=%s" %
              (angle, actual, normal[0].cpu().tolist()), flush=True)
        if angle == 0.0:
            relative_q = quat_mul(quat_conjugate(robot.data.body_quat_w[:, parent_id]), robot.data.body_quat_w[:, palm_id])
            expected_q = targets.new_tensor(report["zero_parent_to_palm_quat_wxyz"]).unsqueeze(0)
            assert float((relative_q * expected_q).sum(-1).abs().min()) > 0.999
    for normal in samples[1:3]:
        change = float(torch.acos((samples[0] * normal).sum().clamp(-1, 1)))
        assert change > 0.5, ("Wrist rotation did not sufficiently tilt the paddle", change)
    assert max_attachment_error < 0.005, max_attachment_error
    paddle_path = robot.root_physx_view.link_paths[0][paddle_id]
    material = robot._physics_sim_view.create_rigid_body_view(paddle_path).get_material_properties()
    assert bool(torch.allclose(material[..., 2], torch.full_like(material[..., 2], 0.8), atol=1e-5))
    print("[PASS] 38 movable joints; wrist driven separately; limits +/-pi/2", flush=True)
    print("[PASS] Wrist changes paddle normal; fixed attachment error %.6fm; restitution 0.8" % max_attachment_error, flush=True)
    report["asset_check_passed"] = True
    report_path = project / "legged_lab/assets/unitree/G1_TT_WRIST/wrist_joint_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("G1 WRIST ASSET CHECK PASSED", flush=True)
    if gui:
        sim.set_camera_view(eye=(-2.8, -1.3, 1.2), target=(-1.4, -0.25, 0.65))
        print("[INFO] GUI preview: wrist oscillates; close window or Ctrl+C to stop.", flush=True)
        while app.is_running():
            advance(0.8 * math.sin(sim.current_time * 1.5))


def patch_registry(source):
    def names(value):
        return [node.args[0].value for node in ast.walk(ast.parse(value))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "register" and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)]
    before = names(source)
    if START in source or END in source:
        if source.count(START) != 1 or source.count(END) != 1:
            raise RuntimeError("Unexpected wrist registration markers")
        first = source.index(START)
        last = source.index(END) + len(END)
        if source[last:last+1] == "\n":
            last += 1
        result = source[:first] + REGISTRATION + source[last:]
    else:
        if {"g1_tt_wrist", "g1_tt_wrist_eval"}.intersection(before):
            raise RuntimeError("Unmarked wrist tasks already registered")
        result = source.rstrip() + "\n\n" + REGISTRATION
    compile(result, "envs/__init__.py", "exec")
    after = names(result)
    new = {"g1_tt_wrist", "g1_tt_wrist_eval"}
    assert [name for name in before if name not in new] == [name for name in after if name not in new]
    assert all(after.count(name) == 1 for name in new)
    return result


def install_task(project):
    entry = project / "legged_lab/envs/__init__.py"
    original = entry.read_text(encoding="utf-8")
    updated = patch_registry(original)
    sources = {
        project / "legged_lab/envs/g1_tt_wrist/__init__.py": "",
        project / "legged_lab/envs/g1_tt_wrist/g1_tt_wrist_config.py": TASK_CONFIG,
        project / "legged_lab/envs/g1_tt_wrist/g1_tt_wrist_env.py": TASK_ENV,
    }
    for path, content in sources.items():
        compile(content, str(path), "exec")
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise RuntimeError("Existing different variant source preserved: " + str(path))
    for path, content in sources.items():
        write_new(path, content)
    if updated != original:
        backup = entry.with_name(entry.name + ".wrist_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".bak")
        shutil.copy2(entry, backup)
        entry.write_text(updated, encoding="utf-8")
        print("Backup:", backup, flush=True)
    print("[PASS] Registered g1_tt_wrist / g1_tt_wrist_eval", flush=True)


def check_task(project):
    import torch
    import legged_lab.envs
    from legged_lab.utils import task_registry
    from legged_lab.envs.g1_tt.g1_tt_config import G1_CONTROL_JOINTS, G1_HAND_JOINTS
    from rsl_rl.runners import OnPolicyPredictorRegressionRunner
    env = None
    try:
        for task in ("g1_tt_wrist", "g1_tt_wrist_eval"):
            cfg, agent = task_registry.get_cfgs(task)
            assert cfg.robot.num_actions == 24 and cfg.robot.num_joints == 38
            assert cfg.actions.joint_names == G1_CONTROL_JOINTS + [JOINT]
            assert cfg.observations.joint_names == cfg.actions.joint_names
            assert agent.experiment_name == "g1_table_tennis_wrist" and agent.resume is False
        original_cfg, original_agent = task_registry.get_cfgs("g1_tt")
        assert original_cfg.robot.num_actions == 23 and original_cfg.robot.num_joints == 37
        assert original_agent.experiment_name == "g1_table_tennis"
        cfg, agent = task_registry.get_cfgs("g1_tt_wrist")
        cfg, agent = deepcopy(cfg), deepcopy(agent)
        cfg.scene.num_envs = 2
        cfg.scene.seed = agent.seed
        cfg.noise.add_noise = True
        cfg.domain_rand.action_delay.enable = False
        cfg.domain_rand.perception_delay.enable = False
        cfg.domain_rand.events.push_robot = None
        cfg.domain_rand.events.add_base_mass = None
        cfg.domain_rand.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        cfg.domain_rand.events.reset_base.params["velocity_range"] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        cfg.domain_rand.events.reset_locomotion_joints.params["position_range"] = (1.0, 1.0)
        cfg.domain_rand.events.reset_manipulation_joints.params["position_range"] = (0.0, 0.0)
        assert cfg.scene.robot.spawn.rigid_props.disable_gravity is False
        assert not cfg.scene.robot.spawn.articulation_props.fix_root_link
        env = task_registry.get_task_class("g1_tt_wrist")(cfg, headless=True)
        assert env.robot.num_joints == 38 and env.num_actions == 24
        assert {env.robot.joint_names[i] for i in env.held_joint_ids} == set(G1_HAND_JOINTS)
        wrist = env.wrist_joint_ids[0]
        assert env.action_joint_ids[-1] == wrist and wrist not in env.held_joint_ids
        obs, extras = env.get_observations()
        assert obs.shape == (2, 450) and extras["observations"]["critic"].shape == (2, 535)
        runner = OnPolicyPredictorRegressionRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        with torch.no_grad():
            actions = runner.alg.policy.act_inference(obs)
            values = runner.alg.policy.evaluate(extras["observations"]["critic"])
            pred = runner._predictor(torch.zeros((2, 15), device=runner.device))
            assert actions.shape == (2, 24) and values.shape == (2, 1) and pred.shape == (2, 3)
            assert all(bool(torch.isfinite(value).all()) for value in (actions, values, pred))
            for _ in range(24):
                actions = torch.zeros((2, 24), device=env.device)
                actions[:, -1] = 1.0  # default + 0.25 rad, within the new limits
                obs, reward, _, extras = env.step(actions)
                runner._record_ball_positions()
                assert bool(torch.isfinite(obs).all()) and bool(torch.isfinite(reward).all())
                assert bool(torch.isfinite(extras["observations"]["critic"]).all())
        assert torch.allclose(env.robot.data.joint_pos_target[:, wrist],
                              env.robot.data.default_joint_pos[:, wrist] + env.action_scale)
        # Reset only environment 0; ensure the new wrist's actual state and target reset.
        ids = torch.tensor([0], device=env.device)
        disturbed = env.robot.data.default_joint_pos[ids][:, [wrist]] + 0.4
        env.robot.write_joint_state_to_sim(disturbed, torch.zeros_like(disturbed), joint_ids=[wrist], env_ids=ids)
        env.reset(ids)
        assert torch.allclose(env.robot.data.joint_pos[ids][:, [wrist]],
                              env.robot.data.default_joint_pos[ids][:, [wrist]], atol=1e-5)
        assert torch.allclose(env.robot.data.joint_pos_target[ids][:, [wrist]],
                              env.robot.data.default_joint_pos[ids][:, [wrist]])
        assert runner.current_learning_iteration == 0 and runner.tot_timesteps == 0
        print("[PASS] Original G1 tasks retained; new task has 24 actions, 38 joints, 14 held fingers", flush=True)
        print("[PASS] Actor 450->24, critic 535->1, predictor 15->3", flush=True)
        print("[PASS] 24 control steps with noise; wrist action and selected-environment reset checked", flush=True)
        print("[PASS] No training and no old checkpoint loaded", flush=True)
        print("G1 WRIST TASK CHECK PASSED", flush=True)
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("asset", "task"), default="asset")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if args.gui and args.stage != "asset":
        parser.error("GUI preview belongs to --stage asset")
    project = Path.cwd()
    if not (project / "legged_lab/assets/unitree/g1_tt.py").is_file():
        parser.error("Run from /workspace/projects/PACE-ICRA2026")
    sys.path.insert(0, str(project))
    if args.stage == "task":
        report_path = project / "legged_lab/assets/unitree/G1_TT_WRIST/wrist_joint_report.json"
        if not report_path.is_file():
            parser.error("Run --stage asset and confirm its check first")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not report.get("asset_check_passed"):
            parser.error("Asset check has not passed; run --stage asset first")
    snapshot = protected_snapshot(project)
    # Task registration must be written before importing legged_lab.envs.
    if args.stage == "task":
        install_task(project)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=not args.gui).app
    try:
        if args.stage == "asset":
            report = build_asset(project)
            check_asset(project, report, args.gui, app)
            report["asset_check_passed"] = True
            path = project / "legged_lab/assets/unitree/G1_TT_WRIST/wrist_joint_report.json"
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        else:
            check_task(project)
        assert protected_snapshot(project) == snapshot, "Original source/asset changed"
        print("[PASS] Protected original G1/T1 files unchanged", flush=True)
    except KeyboardInterrupt:
        print("[INFO] Preview interrupted.", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
