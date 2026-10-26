"""Preview actual bilateral ball flights with a neutral right-paddle robot; no learning."""
import argparse
from copy import deepcopy
import hashlib
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
    import isaaclab.utils.math as math_utils
    import legged_lab.envs
    from legged_lab.utils.task_registry import task_registry

    manifest = json.loads((ROOT / "tools/h1_2/bilateral_install_report.json").read_text())
    check = json.loads((ROOT / "tools/h1_2/bilateral_check_report.json").read_text())
    assert check["bilateral_interface_passed"], "BILATERAL INTERFACE CHECK MUST PASS FIRST"
    for name, sha in {**manifest["protected_sha256"], **manifest["new_files_sha256"]}.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, "SOURCE CHANGED: " + name
    cfg, _ = task_registry.get_cfgs("h1_2_tt_bilateral")
    cfg = deepcopy(cfg)
    cfg.scene.num_envs = 1
    cfg.scene.seed = 42
    cfg.device = args.device
    cfg.noise.add_noise = False
    cfg.domain_rand.action_delay.enable = False
    cfg.domain_rand.perception_delay.enable = False
    cfg.domain_rand.events.push_robot = None
    cfg.domain_rand.events.add_base_mass = None
    # A neutral pose with centered heading to inspect the two incoming sides.
    # Keep free-base physics, gravity, real aerodynamics and the formal sampler.
    cfg.domain_rand.events.reset_base.params["pose_range"] = {
        "x": (-0.35, -0.35), "y": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    cfg.domain_rand.events.reset_base.params["velocity_range"] = {
        k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.domain_rand.events.reset_locomotion_joints.params["position_range"] = (1.0, 1.0)
    cfg.domain_rand.events.reset_manipulation_joints.params["position_range"] = (0.0, 0.0)
    for event in (cfg.domain_rand.events.reset_locomotion_joints, cfg.domain_rand.events.reset_manipulation_joints):
        if "velocity_range" in event.params:
            event.params["velocity_range"] = (0.0, 0.0)

    with torch.no_grad():
        env = task_registry.get_task_class("h1_2_tt_bilateral")(cfg, headless=args.headless)
        assert not env.robot.is_fixed_base and env.num_actions == 27
        assert cfg.scene.robot.spawn.rigid_props.disable_gravity is False
        env.sim.set_camera_view([-3.5, -3.2, 2.2], [-0.65, 0.0, 0.95])
        actions = torch.zeros((1, 27), device=env.device)
        frames = round(args.seconds / env.step_dt)
        log_every = max(1, round(2.0 / env.step_dt))
        wall_start = time.perf_counter()
        completed = non_timeout_resets = timeout_resets = 0
        min_height, max_tilt = float("inf"), 0.0
        # Counters remain monotonic even when a full robot reset restarts core counters.
        generation = int(env.serve_count_by_side.sum().item())
        previous_ball = (env.ball.data.root_pos_w - env.scene.env_origins)[0].clone()
        sampled_at_start = env.serve_count_by_side[0].clone()
        initial_side = int(env.serve_side[0].item())
        crossing_records = []
        crossed = False
        strike_x = -1.70
        print("[PREVIEW] Actual ball flights, neutral actions, free base, gravity; no learned policy", flush=True)
        print("[PREVIEW] Sampled sides alternate; body-relative crossing side is diagnostic, not a stroke label", flush=True)
        print("[SERVE] side:", "positive Y" if env.serve_side[0] > 0 else "negative Y", flush=True)
        for step in range(frames):
            if not app.is_running():
                break
            obs, reward, done, extras = env.step(actions)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all(), "NONFINITE PREVIEW STATE"
            now_generation = int(env.serve_count_by_side.sum().item())
            ball = (env.ball.data.root_pos_w - env.scene.env_origins)[0].clone()
            if now_generation != generation:
                # Never interpret a teleport on reset as crossing the strike plane.
                generation = now_generation
                crossed = False
                print("[SERVE] side:", "positive Y" if env.serve_side[0] > 0 else "negative Y", flush=True)
            elif not crossed and previous_ball[0] > strike_x and ball[0] <= strike_x:
                fraction = ((previous_ball[0] - strike_x) / (previous_ball[0] - ball[0]).clamp_min(1e-8)).clamp(0, 1)
                crossing = previous_ball + fraction * (ball - previous_ball)
                body_relative = math_utils.quat_rotate_inverse(
                    env.robot.data.root_link_quat_w,
                    (crossing + env.scene.env_origins[0] - env.robot.data.root_link_pos_w[0]).unsqueeze(0),
                )[0]
                record = {
                    "serve_side": int(env.serve_side[0].item()),
                    "crossing_xyz_env": crossing.tolist(),
                    "crossing_body_y": float(body_relative[1].item()),
                }
                crossing_records.append(record)
                crossed = True
                print(f"[CROSSING] x={strike_x:.2f} env_y={crossing[1].item():+.3f} z={crossing[2].item():.3f} body_y={body_relative[1].item():+.3f}", flush=True)
            previous_ball = ball
            timeouts = extras["time_outs"].bool()
            non_timeout_resets += int((done.bool() & ~timeouts).sum().item())
            timeout_resets += int((done.bool() & timeouts).sum().item())
            height = (env.robot.data.root_link_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
            tilt = torch.acos((-env.robot.data.projected_gravity_b[0, 2]).clamp(-1.0, 1.0)).item()
            min_height, max_tilt = min(min_height, height), max(max_tilt, tilt)
            completed = step + 1
            if step % log_every == 0 or completed == frames:
                print(f"[PREVIEW] t={completed*env.step_dt:.2f}s base_z={height:.3f} tilt={tilt:.3f} non_timeout_resets={non_timeout_resets} timeout_resets={timeout_resets}", flush=True)
            if not args.headless:
                remaining = wall_start + completed * env.step_dt - time.perf_counter()
                if remaining > 0:
                    time.sleep(min(remaining, env.step_dt))

    # Include the initial serve, and any newly generated serves during the preview.
    counts = (env.serve_count_by_side[0] - sampled_at_start).tolist()
    counts[0 if initial_side < 0 else 1] += 1
    report = {
        "task": "h1_2_tt_bilateral", "trained_policy_loaded": False,
        "neutral_actions": True, "fixed_base": False, "gravity_disabled": False,
        "simulation_seconds": completed * env.step_dt, "requested_seconds": args.seconds,
        "preview_serve_counts_negative_positive": counts,
        "crossing_plane_x_env": strike_x, "observed_crossings": crossing_records,
        "negative_body_y_crossings": sum(r["crossing_body_y"] < 0 for r in crossing_records),
        "positive_body_y_crossings": sum(r["crossing_body_y"] > 0 for r in crossing_records),
        "minimum_observed_base_height": min_height if completed else None,
        "maximum_observed_tilt_rad": max_tilt,
        "non_timeout_resets": non_timeout_resets, "timeout_resets": timeout_resets,
        "scope": "neutral GUI preview and interpolated ball-plane crossings; no table-contact, hit-success or forehand/backhand skill acceptance",
    }
    out = ROOT / "tools/h1_2/bilateral_preview_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("[PREVIEW] Body-Y negative/positive crossings:", report["negative_body_y_crossings"], report["positive_body_y_crossings"], flush=True)
    print("[PREVIEW] Non-timeout/timeout resets:", non_timeout_resets, timeout_resets, flush=True)
    print("Report:", out, flush=True)
    print("H1-2 BILATERAL PREVIEW FINISHED", flush=True)
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    try:
        if env is not None:
            env.close()
    finally:
        app.close()
