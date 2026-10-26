"""Check registered H1-2 task interfaces with two environments, without learning."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import traceback

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
    from legged_lab.assets.unitree.h1_2_tt import H1_2_CONTROL_JOINTS
    from rsl_rl.runners import OnPolicyPredictorRegressionRunner

    manifest_path = ROOT / "tools/h1_2/task_install_report.json"
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["protected_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f"PROTECTED FILE CHANGED: {name}"
    for task, actions, joints, experiment in (
        ("g1_tt", 23, 37, "g1_table_tennis"),
        ("g1_tt_wrist", 24, 38, "g1_table_tennis_wrist"),
    ):
        old_cfg, old_agent = task_registry.get_cfgs(task)
        assert old_cfg.robot.num_actions == actions and old_cfg.robot.num_joints == joints
        assert old_agent.experiment_name == experiment
    for task in ("t1_tt", "t1_tt_eval", "g1_tt_eval", "g1_tt_wrist_eval"):
        assert task in task_registry.task_classes, f"ORIGINAL TASK MISSING: {task}"
    for task in ("h1_2_tt", "h1_2_tt_eval"):
        cfg, agent = task_registry.get_cfgs(task)
        assert cfg.robot.num_actions == 27 and cfg.robot.num_joints == 27
        assert cfg.actions.joint_names == H1_2_CONTROL_JOINTS
        assert cfg.observations.joint_names == H1_2_CONTROL_JOINTS
        assert agent.experiment_name == "h1_2_table_tennis" and agent.resume is False
        assert cfg.reward.paddel_head_too_near is None
        assert cfg.scene.robot.spawn.rigid_props.disable_gravity is False
        assert cfg.scene.robot.spawn.articulation_props.fix_root_link is False

    cfg, agent = task_registry.get_cfgs("h1_2_tt")
    cfg, agent = deepcopy(cfg), deepcopy(agent)
    cfg.scene.num_envs = 2
    cfg.device = args.device
    agent.device = args.device
    cfg.scene.seed = agent.seed
    cfg.noise.add_noise = True
    cfg.domain_rand.action_delay.enable = False
    cfg.domain_rand.perception_delay.enable = False
    cfg.domain_rand.events.push_robot = None
    cfg.domain_rand.events.add_base_mass = None
    cfg.domain_rand.events.reset_base.params["pose_range"] = {key: (0.0, 0.0) for key in ("x", "y", "yaw")}
    cfg.domain_rand.events.reset_base.params["velocity_range"] = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.domain_rand.events.reset_locomotion_joints.params["position_range"] = (1.0, 1.0)
    cfg.domain_rand.events.reset_manipulation_joints.params["position_range"] = (0.0, 0.0)
    assert cfg.robot.actor_obs_history_length == 5 and cfg.robot.critic_obs_history_length == 5

    with torch.no_grad():
        print("[CHECK] Creating two H1-2 environments with free bases and gravity", flush=True)
        env = task_registry.get_task_class("h1_2_tt")(cfg, headless=args.headless)
        assert env.num_actions == 27 and env.robot.num_joints == 27
        assert not env.robot.is_fixed_base
        assert env.held_joint_ids == [] and len(env.wrist_joint_ids) == 6
        assert len(set(env.action_joint_ids)) == 27
        assert torch.isfinite(env._h1_2_measured_paddle_offset_b).all()
        sample_target = torch.zeros((2, 3), device=env.device)
        base_target = env._h1_2_base_target(sample_target)
        base_height = base_target[:, 2] + env.table.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
        assert torch.allclose(base_height, torch.full_like(base_height, cfg.reference_base_height), atol=1e-6)

        obs, extras = env.get_observations()
        assert obs.shape == (2, 495) and extras["observations"]["critic"].shape == (2, 580)
        print("[CHECK] Constructing new policy and predictor", flush=True)
        runner = OnPolicyPredictorRegressionRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        action = runner.alg.policy.act_inference(obs)
        value = runner.alg.policy.evaluate(extras["observations"]["critic"])
        prediction = runner._predictor(torch.zeros((2, 15), device=runner.device))
        assert action.shape == (2, 27) and value.shape == (2, 1) and prediction.shape == (2, 3)
        assert all(torch.isfinite(v).all() for v in (action, value, prediction))
        env.update_prediction(prediction)

        wrist_actions = {
            "right_wrist_roll_joint": 0.4,
            "right_wrist_pitch_joint": -0.2,
            "right_wrist_yaw_joint": 0.4,
        }
        for step in range(24):
            actions = torch.zeros((2, 27), device=env.device)
            for name, amplitude in wrist_actions.items():
                actions[:, env.action_joint_names.index(name)] = amplitude
            obs, rewards, dones, extras = env.step(actions)
            runner._record_ball_positions()
            assert obs.shape == (2, 495) and extras["observations"]["critic"].shape == (2, 580)
            assert all(torch.isfinite(v).all() for v in (obs, rewards, extras["observations"]["critic"], env.robot.data.joint_pos, env.paddle_normal))
            assert torch.allclose(env.paddle_normal.norm(dim=-1), torch.ones(2, device=env.device), atol=1e-5)
            if step in (0, 23):
                print(f"[CHECK] Noisy control step {step + 1}/24", flush=True)
        for name, amplitude in wrist_actions.items():
            jid = env.robot.find_joints(name)[0][0]
            expected = env.robot.data.default_joint_pos[:, jid] + amplitude * env.action_scale
            assert torch.allclose(env.robot.data.joint_pos_target[:, jid], expected, atol=1e-6), f"WRIST ACTION MAPPING FAILED: {name}"

        print("[CHECK] Resetting only environment 0: six wrist states and all control targets", flush=True)
        chosen = torch.tensor([0], device=env.device)
        wrists = env.wrist_joint_ids
        other_state = env.robot.data.joint_pos[1].clone()
        other_target = env.robot.data.joint_pos_target[1].clone()
        disturbed = env.robot.data.default_joint_pos[chosen][:, wrists] + 0.1
        env.robot.write_joint_state_to_sim(disturbed, torch.zeros_like(disturbed), joint_ids=wrists, env_ids=chosen)
        env.robot.set_joint_position_target(disturbed, joint_ids=wrists, env_ids=chosen)
        env.reset(chosen)
        assert torch.allclose(env.robot.data.joint_pos[chosen][:, wrists], env.robot.data.default_joint_pos[chosen][:, wrists], atol=1e-5)
        assert torch.allclose(env.robot.data.joint_vel[chosen][:, wrists], env.robot.data.default_joint_vel[chosen][:, wrists], atol=1e-5)
        assert torch.allclose(env.robot.data.joint_pos_target[chosen][:, wrists], env.robot.data.default_joint_pos[chosen][:, wrists], atol=1e-6)
        assert torch.allclose(env.robot.data.joint_pos[1], other_state, atol=1e-6), "RESET CHANGED OTHER ENVIRONMENT STATE"
        assert torch.allclose(env.robot.data.joint_pos_target[1], other_target, atol=1e-6), "RESET CHANGED OTHER ENVIRONMENT TARGET"
        env.add_noise = False
        obs, extras = env.get_observations()
        assert torch.isfinite(obs).all() and torch.isfinite(extras["observations"]["critic"]).all()
        assert runner.current_learning_iteration == 0 and runner.tot_timesteps == 0

    report = {
        "task_check_passed": True, "num_envs": 2, "control_steps": 24,
        "num_actions": 27, "num_joints": 27,
        "actor_input": 495, "actor_output": 27, "critic_input": 580, "critic_output": 1,
        "predictor_input": 15, "predictor_output": 3,
        "reference_geometry": env._h1_2_reference_geometry,
        "measured_paddle_offset_b": env._h1_2_measured_paddle_offset_b[0].tolist(),
        "head_distance_reward_disabled": True,
        "protected_files_verified": True, "learning_iterations": 0, "learned_steps": 0,
        "scope": "task interface only; no learned H1-2 performance or long-horizon standing stability claim",
    }
    for name, digest in manifest["protected_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f"PROTECTED FILE CHANGED: {name}"
    out = ROOT / "legged_lab/assets/unitree/H1_2_TT/task_check_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("[PASS] 27 actions/joints, actor 495->27, critic 580->1, predictor 15->3", flush=True)
    print("[PASS] Noisy steps, native wrist actions and selected-environment reset", flush=True)
    print("[PASS] Original files verified; new policy initialized; zero learning iterations", flush=True)
    print("Reference base height:", cfg.reference_base_height, flush=True)
    print("Reference paddle offset (base frame):", report["measured_paddle_offset_b"], flush=True)
    print("Report:", out, flush=True)
    print("H1-2 TASK CHECK PASSED", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    try:
        if env is not None:
            env.close()
    finally:
        app.close()
