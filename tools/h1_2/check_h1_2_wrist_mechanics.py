"""Check H1-2 wrist drives in a temporary fixed-base, gravity-free fixture."""
import argparse
import importlib.util
import json
from pathlib import Path
import traceback
from datetime import datetime, timezone, timedelta

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--hold-gui", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.hold_gui and args.headless:
    parser.error("--hold-gui requires GUI mode")
app = AppLauncher(args).app

try:
    import torch
    import isaaclab.sim as sim_utils
    import isaaclab.utils.math as math_utils
    from isaaclab.assets import Articulation

    root = Path("/workspace/projects/PACE-ICRA2026")
    config_path = root / "legged_lab/assets/unitree/h1_2_tt.py"
    spec = importlib.util.spec_from_file_location("pace_h1_2_asset_check", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    original = config.H1_2_TT_CFG
    assert original.spawn.articulation_props.fix_root_link is False
    assert original.spawn.rigid_props.disable_gravity is False

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 240.0, device=args.device))
    sim.set_camera_view([2.2, -2.6, 2.2], [0.0, 0.0, 1.4])
    light = sim_utils.DomeLightCfg(intensity=2500.0)
    light.func("/World/Light", light)
    fixture = original.copy()
    fixture.prim_path = "/World/H1_2"
    fixture.init_state.pos = (0.0, 0.0, 1.2)
    fixture.spawn.articulation_props.fix_root_link = True
    fixture.spawn.rigid_props.disable_gravity = True
    robot = Articulation(fixture)
    sim.reset()
    assert robot.is_initialized, "ARTICULATION INITIALIZATION FAILED"
    assert robot.is_fixed_base, "MECHANICAL FIXTURE BASE NOT FIXED"
    assert robot.num_joints == 27
    ids, names = robot.find_joints(config.H1_2_CONTROL_JOINTS, preserve_order=True)
    assert names == config.H1_2_CONTROL_JOINTS and len(set(ids)) == 27
    controlled = [n for actuator in robot.actuators.values() for n in actuator.joint_names]
    assert len(controlled) == 27 and set(controlled) == set(names), "ACTUATOR COVERAGE ERROR"
    for jid, name in zip(ids, names):
        expected = config.H1_2_JOINT_SPECS[name]
        limits = robot.data.joint_pos_limits[0, jid]
        assert abs(limits[0].item() - expected["lower"]) < 1e-4, f"LOWER LIMIT MISMATCH: {name}"
        assert abs(limits[1].item() - expected["upper"]) < 1e-4, f"UPPER LIMIT MISMATCH: {name}"
        assert abs(robot.data.joint_effort_limits[0, jid].item() - expected["effort"]) < 1e-4, f"EFFORT LIMIT MISMATCH: {name}"
        assert abs(robot.data.joint_vel_limits[0, jid].item() - expected["velocity"]) < 1e-4, f"VELOCITY LIMIT MISMATCH: {name}"
    wrist_id = robot.find_bodies(config.H1_2_PADDLE_PARENT_BODY_NAME)[0][0]
    paddle_id = robot.find_bodies(config.H1_2_PADDLE_BODY_NAME)[0][0]
    offset = torch.tensor(config.H1_2_PADDLE_PARENT_OFFSET, device=robot.device).unsqueeze(0)
    local_normal = torch.tensor(config.H1_2_PADDLE_NORMAL, device=robot.device).unsqueeze(0)
    dt = sim.get_physics_dt()
    max_anchor = 0.0
    max_rotation = 0.0
    results = []

    def snapshot():
        wp = robot.data.body_link_pos_w[:, wrist_id]
        wq = robot.data.body_link_quat_w[:, wrist_id]
        pp = robot.data.body_link_pos_w[:, paddle_id]
        pq = robot.data.body_link_quat_w[:, paddle_id]
        wq = wq / wq.norm(dim=-1, keepdim=True)
        pq = pq / pq.norm(dim=-1, keepdim=True)
        anchor = wp + math_utils.quat_apply(wq, offset)
        error = torch.linalg.vector_norm(pp - anchor, dim=-1).max().item()
        rotation = 2.0 * torch.acos((wq * pq).sum(-1).abs().clamp(0.0, 1.0)).max().item()
        normal = math_utils.quat_apply(pq, local_normal)
        return error, rotation, pq.clone(), normal.clone(), pp.clone()

    def advance(target, seconds=2.0):
        global max_anchor, max_rotation
        start = robot.data.joint_pos.clone()
        for step in range(round(seconds / dt)):
            assert app.is_running(), "APPLICATION CLOSED BEFORE CHECK FINISHED"
            alpha = min((step + 1) * dt / 0.5, 1.0)
            robot.set_joint_position_target(start + (target - start) * alpha)
            robot.write_data_to_sim()
            sim.step(render=not args.headless)
            robot.update(dt)
            assert torch.isfinite(robot.data.joint_pos).all(), "NONFINITE JOINT POSITION"
            assert torch.isfinite(robot.data.joint_vel).all(), "NONFINITE JOINT VELOCITY"
            error, rotation, _, _, _ = snapshot()
            assert torch.isfinite(robot.data.body_link_pos_w).all(), "NONFINITE BODY POSITION"
            max_anchor = max(max_anchor, error)
            max_rotation = max(max_rotation, rotation)
            assert error < 0.002, f"PADDLE ANCHOR ERROR: {error} m"
            assert rotation < 0.01, f"PADDLE ROTATION ERROR: {rotation} rad"

    print("Fixture: fixed base, gravity disabled; formal config remains free with gravity", flush=True)
    print("Moving joints:", robot.num_joints, "Actuator coverage:", len(controlled), flush=True)
    with torch.no_grad():
        default = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(default, robot.data.default_joint_vel.clone())
        robot.reset()
        advance(default)
        for axis, amplitude in (("roll", 0.60), ("pitch", 0.25), ("yaw", 0.60)):
            name = f"right_wrist_{axis}_joint"
            jid = robot.find_joints(name)[0][0]
            limits = robot.data.joint_pos_limits[0, jid]
            assert limits[0].item() < -amplitude and limits[1].item() > amplitude
            advance(default)
            _, _, reference_q, reference_normal, reference_position = snapshot()
            for value in (0.0, -amplitude, amplitude, 0.0):
                target = default.clone()
                target[:, jid] = value
                advance(target)
                actual = robot.data.joint_pos[0, jid].item()
                tracking = abs(actual - value)
                error, rotation, quat, normal, position = snapshot()
                change = (2.0 * torch.acos((quat * reference_q).sum(-1).abs().clamp(0.0, 1.0))).item()
                normal_change = torch.linalg.vector_norm(normal - reference_normal, dim=-1).item()
                center_change = torch.linalg.vector_norm(position - reference_position, dim=-1).item()
                print(f"{name}: target={value:+.3f} actual={actual:+.3f} error={tracking:.6f} pose_change={change:.3f} normal_change={normal_change:.3f} anchor_error={error:.8f}", flush=True)
                assert tracking < 0.04, f"WRIST TRACKING FAILED: {name}"
                if abs(value) > 0.1:
                    assert change > 0.10, f"PADDLE POSE DID NOT FOLLOW: {name}"
                results.append({
                    "joint": name, "target": value, "actual": actual,
                    "tracking_error_rad": tracking, "paddle_pose_change_rad": change,
                    "normal_vector_change": normal_change, "center_displacement_m": center_change,
                    "anchor_error_m": error, "rotation_error_rad": rotation,
                })
        advance(default)
        assert (robot.data.joint_pos - default).abs().max().item() < 0.04, "FINAL RETURN TO DEFAULT FAILED"

    # A rotation around local Y can leave the Y normal unchanged while the paddle pose changes.
    report = {
        "time_asia_shanghai": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "fixture_fixed_base": True, "fixture_gravity_disabled": True,
        "formal_config_fixed_base": False, "formal_config_gravity_disabled": False,
        "joint_names": robot.joint_names, "body_names": robot.body_names,
        "actuator_joint_coverage": controlled,
        "max_anchor_error_m": max_anchor, "max_relative_rotation_error_rad": max_rotation,
        "tests": results,
        "scope": "isolated wrist tracking and paddle attachment; ground support and task validation pending",
    }
    out = root / "legged_lab/assets/unitree/H1_2_TT/wrist_mechanical_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("Max paddle anchor error (m):", max_anchor, flush=True)
    print("Report:", out, flush=True)
    print("H1-2 WRIST MECHANICAL CHECK PASSED", flush=True)
    if args.hold_gui:
        while app.is_running():
            robot.set_joint_position_target(default)
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    app.close()
