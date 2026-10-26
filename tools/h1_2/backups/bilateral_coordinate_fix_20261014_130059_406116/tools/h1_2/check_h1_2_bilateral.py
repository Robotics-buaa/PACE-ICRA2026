"""Audit serve geometry, observation interfaces and selected resets; no learning."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path("/workspace/projects/PACE-ICRA2026")
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
env = None

try:
    import torch
    import legged_lab.envs
    from legged_lab.utils.task_registry import task_registry
    from rsl_rl.runners import OnPolicyPredictorRegressionRunner
    manifest = json.loads((ROOT / "tools/h1_2/bilateral_install_report.json").read_text())
    def verify_files():
        for name, digest in manifest["protected_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        for name, digest in manifest["new_files_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    verify_files()
    for name, vy, reset_y in (
        ("h1_2_tt", (-0.8, 0.4), (-0.6, 0.8)),
        ("h1_2_tt_eval", (-0.6, 0.2), (0.3, 0.4)),
    ):
        old, old_agent = task_registry.get_cfgs(name)
        assert tuple(old.ball.ball_speed_y_range) == vy, "BASELINE SERVE CHANGED"
        assert tuple(old.domain_rand.events.reset_base.params["pose_range"]["y"]) == reset_y
        assert old_agent.experiment_name == "h1_2_table_tennis"
    for name in ("h1_2_tt_bilateral", "h1_2_tt_bilateral_eval"):
        cfg, agent = task_registry.get_cfgs(name)
        assert cfg.ball.ball_reset_repeat == 1 and tuple(cfg.ready_base_xy) == (-1.8, 0.0)
        assert tuple(cfg.ball.ball_speed_y_range) == (-0.65, 0.65)
        assert tuple(cfg.domain_rand.events.reset_base.params["pose_range"]["y"]) == (-0.05, 0.05)
        assert agent.experiment_name == "h1_2_table_tennis_bilateral" and not agent.resume
        assert cfg.robot.num_actions == 27
    cfg, agent = task_registry.get_cfgs("h1_2_tt_bilateral")
    cfg, agent = deepcopy(cfg), deepcopy(agent)
    cfg.scene.num_envs = 4
    cfg.scene.seed = 42
    cfg.device = agent.device = args.device
    cfg.domain_rand.action_delay.enable = False
    cfg.domain_rand.perception_delay.enable = False
    cfg.domain_rand.events.push_robot = None
    cfg.domain_rand.events.add_base_mass = None
    # Deterministic neutral fixture for interface checks, not formal train resets.
    cfg.domain_rand.events.reset_base.params["pose_range"] = {k: (0.0, 0.0) for k in ("x", "y", "yaw")}
    cfg.domain_rand.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.domain_rand.events.reset_locomotion_joints.params["position_range"] = (1.0, 1.0)
    cfg.domain_rand.events.reset_manipulation_joints.params["position_range"] = (0.0, 0.0)
    with torch.no_grad():
        env = task_registry.get_task_class("h1_2_tt_bilateral")(cfg, headless=args.headless)
        assert env.num_actions == 27 and not env.robot.is_fixed_base
        assert cfg.scene.robot.spawn.rigid_props.disable_gravity is False
        ids = torch.arange(4, device=env.device)
        records = []
        lands = []
        velocities = []
        for i in range(256):
            if i:
                env.reset_ball(ids)
            # Cached root writes, then original perception and prediction code.
            env.compute_perception()
            env.compute_paddle_touch()
            env.compute_intermediate_values()
            assert torch.allclose(env.pos_pred_before, env.serve_apex, atol=2e-5), "SAMPLER/PACE FORECAST MISMATCH"
            assert torch.allclose(env.serve_side, torch.sign(env.serve_apex[:, 1]))
            table_xy = env.table.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
            base_xy = env._h1_2_base_target(env.serve_apex)[:, :2] + table_xy
            residual_y = env.serve_apex[:, 1] - base_xy[:, 1]
            assert torch.all(residual_y * env.serve_side > 0.16), "TARGET STILL PLACES BOTH SERVES ON ONE BODY SIDE"
            assert torch.all(base_xy[:, 1].abs() <= cfg.bilateral_base_y_limit + 1e-6)
            records.append(env.serve_apex.clone())
            lands.append(env.serve_landing_xy.clone())
            velocities.append(env.reset_ball_state_buf[:, 7:10].clone())
            if i in (0, 127, 255):
                print(f"[CHECK] Serve audit {(i+1)*4}/1024", flush=True)
        assert torch.equal(env.serve_count_by_side, torch.full_like(env.serve_count_by_side, 128))
        apex = torch.cat(records)
        landing = torch.cat(lands)
        velocity = torch.cat(velocities)
        assert torch.all(velocity[:, 0] < 0), "BALLS MUST STILL TRAVEL TOWARD THE ROBOT"
        assert torch.all(apex[:, 1].abs() >= 0.22) and torch.all(apex[:, 1].abs() <= 0.40)
        assert torch.all(apex[:, 0] >= -1.85) and torch.all(apex[:, 0] <= -1.65)
        margin = cfg.bilateral_table_margin
        assert torch.all(landing[:, 0] >= cfg.table.table_own_contact_x[0] + margin)
        assert torch.all(landing[:, 0] <= cfg.table.table_own_contact_x[1] - margin)
        assert torch.all(landing[:, 1] >= cfg.table.table_own_contact_y[0] + margin)
        assert torch.all(landing[:, 1] <= cfg.table.table_own_contact_y[1] - margin)

        # Sampler labels are diagnostics, never an oracle input to actor guidance.
        fixed_prediction = env.serve_apex.clone()
        target = env._h1_2_base_target(fixed_prediction)
        side = env.serve_side.clone()
        env.serve_side *= -1
        assert torch.equal(target, env._h1_2_base_target(fixed_prediction))
        env.serve_side.copy_(side)
        untouched = env.reset_ball_state_buf[1:].clone()
        untouched_counts = env.serve_count_by_side[1:].clone()
        counter = env.ball_reset_counter.clone()
        old_side = env.serve_side[0].clone()
        env.reset_ball(ids[:1])
        assert env.serve_side[0] == -old_side
        assert torch.equal(env.reset_ball_state_buf[1:], untouched)
        assert torch.equal(env.serve_count_by_side[1:], untouched_counts)
        assert torch.equal(env.ball_reset_counter[1:], counter[1:])
        assert env.ball_reset_counter[0] == counter[0] + 1
        env.reset(ids)
        obs, extras = env.get_observations()
        assert obs.shape == (4, 495) and extras["observations"]["critic"].shape == (4, 580)
        runner = OnPolicyPredictorRegressionRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        action = runner.alg.policy.act_inference(obs)
        value = runner.alg.policy.evaluate(extras["observations"]["critic"])
        prediction = runner._predictor(torch.zeros((4, 15), device=runner.device))
        assert action.shape == (4, 27) and value.shape == (4, 1) and prediction.shape == (4, 3)
        env.update_prediction(prediction)
        for _ in range(24):
            obs, reward, done, extras = env.step(torch.zeros((4, 27), device=env.device))
            assert obs.shape == (4, 495) and extras["observations"]["critic"].shape == (4, 580)
            assert all(torch.isfinite(v).all() for v in (obs, reward, extras["observations"]["critic"]))
        assert runner.current_learning_iteration == 0 and runner.tot_timesteps == 0
    verify_files()
    report = {
        "bilateral_interface_passed": True, "sampled_serves": 1024,
        "negative_y_count": 512, "positive_y_count": 512,
        "apex_min": apex.min(dim=0).values.tolist(), "apex_max": apex.max(dim=0).values.tolist(),
        "landing_min": landing.min(dim=0).values.tolist(), "landing_max": landing.max(dim=0).values.tolist(),
        "velocity_min": velocity.min(dim=0).values.tolist(), "velocity_max": velocity.max(dim=0).values.tolist(),
        "actor": [495, 27], "critic": [580, 1], "predictor": [15, 3],
        "learning_iterations": 0, "control_steps": 24, "baseline_hashes_verified": True,
        "scope": "approximate serve geometry and task interfaces; no measured landing/hit rate, backhand skill or reachability guarantee",
    }
    out = ROOT / "tools/h1_2/bilateral_check_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("[PASS] 1024 serves: negative Y 512; positive Y 512", flush=True)
    print("[PASS] PACE forecast regions inside configured bounds; central stance target allows both body sides", flush=True)
    print("[PASS] Selected reset, finite steps, actor 495->27, critic 580->1, predictor 15->3; no learning", flush=True)
    print("[PASS] Original H1-2/G1/T1 source and assets unchanged", flush=True)
    print("Report:", out, flush=True)
    print("H1-2 BILATERAL INTERFACE CHECK PASSED", flush=True)
finally:
    try:
        if env is not None:
            env.close()
    finally:
        app.close()
