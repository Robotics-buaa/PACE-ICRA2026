"""Preview the real H1-2 table-tennis task with neutral actions and gravity."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path("/workspace/projects/PACE-ICRA2026")
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=30.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.seconds <= 0:
    parser.error("--seconds must be positive")
app = AppLauncher(args).app
env = None

try:
    import torch
    import legged_lab.envs
    from legged_lab.utils.task_registry import task_registry

    check = ROOT / "legged_lab/assets/unitree/H1_2_TT/task_check_report.json"
    assert check.is_file() and json.loads(check.read_text())["task_check_passed"], "TASK INTERFACE CHECK MUST PASS FIRST"
    cfg, _ = task_registry.get_cfgs("h1_2_tt")
    cfg = deepcopy(cfg)
    cfg.scene.num_envs = 1
    cfg.scene.seed = 42
    cfg.device = args.device
    cfg.noise.add_noise = False
    cfg.domain_rand.action_delay.enable = False
    cfg.domain_rand.perception_delay.enable = False
    cfg.domain_rand.events.push_robot = None
    cfg.domain_rand.events.add_base_mass = None
    cfg.domain_rand.events.reset_base.params["pose_range"] = {k: (0.0, 0.0) for k in ("x", "y", "yaw")}
    cfg.domain_rand.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.domain_rand.events.reset_locomotion_joints.params["position_range"] = (1.0, 1.0)
    cfg.domain_rand.events.reset_manipulation_joints.params["position_range"] = (0.0, 0.0)
    for event in (cfg.domain_rand.events.reset_locomotion_joints, cfg.domain_rand.events.reset_manipulation_joints):
        if "velocity_range" in event.params:
            event.params["velocity_range"] = (0.0, 0.0)
    assert cfg.scene.robot.spawn.articulation_props.fix_root_link is False
    assert cfg.scene.robot.spawn.rigid_props.disable_gravity is False

    with torch.no_grad():
        env = task_registry.get_task_class("h1_2_tt")(cfg, headless=args.headless)
        assert not env.robot.is_fixed_base and env.num_actions == 27
        env.sim.set_camera_view([-3.5, -3.2, 2.2], [-1.3, 0.0, 1.0])
        actions = torch.zeros((1, 27), device=env.device)
        non_timeout_resets = 0
        timeout_resets = 0
        minimum_height = float("inf")
        maximum_tilt = 0.0
        frames = round(args.seconds / env.step_dt)
        log_every = max(1, round(2.0 / env.step_dt))
        wall_start = time.perf_counter()
        completed = 0
        print("[PREVIEW] Free base, gravity enabled, 27 neutral actions, no trained policy", flush=True)
        print("[PREVIEW] Reference base height:", cfg.reference_base_height, flush=True)
        for step in range(frames):
            if not app.is_running():
                break
            obs, reward, done, extras = env.step(actions)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all(), "NONFINITE PREVIEW STATE"
            timeouts = extras["time_outs"].bool()
            non_timeout_resets += int((done.bool() & ~timeouts).sum().item())
            timeout_resets += int((done.bool() & timeouts).sum().item())
            height = (env.robot.data.root_link_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
            tilt = torch.acos((-env.robot.data.projected_gravity_b[0, 2]).clamp(-1.0, 1.0)).item()
            minimum_height = min(minimum_height, height)
            maximum_tilt = max(maximum_tilt, tilt)
            completed = step + 1
            if step % log_every == 0 or completed == frames:
                forces = env.contact_sensor.data.net_forces_w[0, env.feet_cfg.body_ids]
                contact = (torch.linalg.vector_norm(forces, dim=-1) > 0.5).tolist()
                print(f"[PREVIEW] t={completed*env.step_dt:.2f}s base_z={height:.3f} tilt_rad={tilt:.3f} feet_contact={contact} non_timeout_resets={non_timeout_resets} timeout_resets={timeout_resets}", flush=True)
            # Pace the GUI so the motion can be inspected at approximately real time.
            if not args.headless:
                remaining = wall_start + completed * env.step_dt - time.perf_counter()
                if remaining > 0:
                    time.sleep(min(remaining, env.step_dt))

    report = {
        "neutral_actions": True, "trained_policy_loaded": False,
        "fixed_base": False, "gravity_disabled": False,
        "simulation_seconds": completed * env.step_dt,
        "requested_seconds": args.seconds,
        "reference_base_height": cfg.reference_base_height,
        "minimum_observed_base_height": minimum_height if completed else None,
        "maximum_observed_tilt_rad": maximum_tilt,
        "non_timeout_resets": non_timeout_resets, "timeout_resets": timeout_resets,
        "scope": "neutral-pose GUI/physics preview; visual acceptance and learned policy performance are separate",
    }
    out = ROOT / "legged_lab/assets/unitree/H1_2_TT/task_preview_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("[PREVIEW] Non-timeout resets:", non_timeout_resets, flush=True)
    print("[PREVIEW] Timeout resets:", timeout_resets, flush=True)
    print("Report:", out, flush=True)
    print("H1-2 TASK PREVIEW FINISHED", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    try:
        if env is not None:
            env.close()
    finally:
        app.close()
