"""Short interface check only: two environments, eight control steps, no model/learning."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path('/workspace/projects/PACE-ICRA2026')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

manifest = json.loads((HERE / 'bundle_manifest.json').read_text())
baseline = json.loads((ROOT / 'tools/h1_2/bilateral_install_report.json').read_text())
protected = {}
for group in (baseline['protected_sha256'], baseline['new_files_sha256'], manifest['required_source_sha256']):
    for rel, digest in group.items():
        assert rel not in protected or protected[rel] == digest, 'CONFLICTING PROTECTION DATA: ' + rel
        protected[rel] = digest
protected.setdefault('legged_lab/envs/__init__.py', hashlib.sha256((ROOT / 'legged_lab/envs/__init__.py').read_bytes()).hexdigest())

def verify():
    for rel, expected in protected.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected, 'PROTECTED SOURCE CHANGED: ' + rel
    for name, expected in manifest['files_sha256'].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, 'STANDING BUNDLE CHANGED: ' + name

verify()
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = None
env = None
report = {'interface_passed': False, 'learning_updates': 0, 'model_loaded': False,
          'stance_performance': 'not tested; eight steps cannot establish standing skill'}
try:
    app = AppLauncher(args).app
    import torch
    import legged_lab.envs
    from legged_lab.utils import task_registry
    from standing_v1.task import register_tasks, height_gate
    original_class = task_registry.get_task_class('h1_2_tt')
    register_tasks()
    assert task_registry.get_task_class('h1_2_tt') is original_class
    cfg, agent_cfg = task_registry.get_cfgs('h1_2_stand_v1')
    cfg = deepcopy(cfg)
    cfg.scene.num_envs = 2
    cfg.device = args.device
    env = task_registry.get_task_class('h1_2_stand_v1')(cfg, args.headless)
    assert not env.robot.is_fixed_base and not cfg.scene.robot.spawn.rigid_props.disable_gravity
    assert not cfg.domain_rand.action_delay.enable and not cfg.noise.add_noise
    assert agent_cfg.experiment_name == 'h1_2_standing_v1' and not agent_cfg.resume
    assert abs(env.max_episode_length * env.step_dt - 20.) <= env.step_dt
    ids = env.action_joint_ids
    lower_names = [f'{s}_{p}_joint' for s in ('left', 'right') for p in ('hip_pitch', 'knee', 'ankle_pitch')]
    lower_ids = [env.robot.joint_names.index(n) for n in lower_names]
    runtime_kp = env.robot.root_physx_view.get_dof_stiffnesses()[0, lower_ids].tolist()
    assert runtime_kp == [200., 200., 60., 200., 200., 60.], 'FORMAL KP CHANGED'
    terms = list(env.reward_manager.active_terms)
    assert set(terms) == set(cfg.reward.__dict__) and len(terms) == 12
    assert not any('future' in n or 'contact_rew' in n for n in terms)

    def assert_goals_and_observations():
        actor, critic = env.compute_current_observations_perception()
        assert actor.shape == (2, 99) and critic.shape == (2, 116)
        assert torch.isfinite(actor).all() and torch.isfinite(critic).all()
        assert torch.equal(actor[:, 87:93], torch.zeros_like(actor[:, 87:93]))
        assert torch.equal(critic[:, 87:93], torch.zeros_like(critic[:, 87:93]))
        assert torch.allclose(critic[:, 96:99], env.paddle_pos * env.obs_scales.ball_pos, atol=1e-6), 'LIVE PADDLE GEOMETRY MISMATCH'
        robot_table = env.robot.data.root_link_pos_w - env.table.data.root_link_pos_w
        expected = (env.stand_goal_table()[:, :2] - robot_table[:, :2]) * env.obs_scales.robot_pos
        assert torch.allclose(actor[:, -3:-1], expected, atol=1e-6)
        assert torch.allclose(critic[:, 103:106], (env.stand_goal_table() - robot_table) * env.obs_scales.robot_pos, atol=1e-6)
        assert torch.equal(env.robot_future_vel, torch.zeros_like(env.robot_future_vel))
        assert torch.allclose(env.robot_future_pos, env.stand_goal_table(), atol=1e-6)
        assert torch.all(env.ball.data.root_pos_w[:, 2] > 9000.)
        limits = env.robot.data.joint_pos_limits
        assert torch.all(env.robot.data.joint_pos_target >= limits[:, :, 0] - 1e-5)
        assert torch.all(env.robot.data.joint_pos_target <= limits[:, :, 1] + 1e-5)

    with torch.no_grad():
        assert_goals_and_observations()
        obs, extras = env.get_observations()
        assert obs.shape == (2, 495) and torch.isfinite(obs).all()
        wrist_action = env.action_joint_names.index('right_wrist_yaw_joint')
        for step in range(8):
            action = torch.zeros((2, 27), device=env.device)
            if step == 3:
                action[0, wrist_action] = 1000.
            obs, reward, done, extras = env.step(action)
            assert obs.shape == (2, 495) and extras['observations']['critic'].shape == (2, 580)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            assert_goals_and_observations()
            if step == 3:
                assert env.stand_projection_delta[0, wrist_action].abs() > 1.
                j = ids[wrist_action]
                assert torch.allclose(env.robot.data.joint_pos_target[0, j], env.robot.data.joint_pos_limits[0, j, 1], atol=1e-5)
        saved_lengths = env.episode_length_buf.clone()
        saved_serves = env.ball_reset_counter.clone()
        try:
            env.episode_length_buf[:] = 0
            env.episode_length_buf[0] = env.max_episode_length
            env.ball_reset_counter[1] = env.max_ball_serve_per_episode + 1
            reset, timeout = env.check_reset()
            assert timeout[0] and not timeout[1] and not reset[1], 'SERVE COUNT STILL ENDS STANDING EPISODE'
        finally:
            env.episode_length_buf[:] = saved_lengths
            env.ball_reset_counter[:] = saved_serves
        untouched_q = env.robot.data.joint_pos[1].clone()
        untouched_pose = env.robot.data.root_link_pos_w[1].clone()
        env.reset(torch.tensor([0], device=env.device))
        assert torch.allclose(env.robot.data.joint_pos[1], untouched_q, atol=1e-6)
        assert torch.allclose(env.robot.data.root_link_pos_w[1], untouched_pose, atol=1e-6)
        assert_goals_and_observations()
        report.update({'interface_passed': True, 'steps': 8, 'num_envs': 2,
                       'runtime_lower_kp': runtime_kp, 'reward_weights': {k: v.weight for k, v in cfg.reward.__dict__.items()},
                       'reference_base_height_m': float(cfg.reference_base_height),
                       'actor_shape': list(obs.shape), 'critic_shape': list(extras['observations']['critic'].shape),
                       'selective_reset_checked': True, 'serve_counter_ignored': True,
                       'original_tasks_and_files_preserved': True, 'height_gate_current': height_gate(env).cpu().tolist()})
    verify()
    print('[STAND-V1] New tasks: h1_2_stand_v1 / h1_2_stand_v1_eval (sidecar registration only)', flush=True)
    print('[STAND-V1] Original lower Kp:', runtime_kp, flush=True)
    print('[STAND-V1] Reference height:', report['reference_base_height_m'], flush=True)
    print('[STAND-V1] Reward weights:', report['reward_weights'], flush=True)
    print('[STAND-V1] actor/critic/actions: 495 / 580 / 27; fixed actor+critic body goals; bounded targets', flush=True)
    print('[STAND-V1] 20s episode independent of serve count; selective reset and protected hashes checked', flush=True)
    print('H1-2 STAND V1 INTERFACE CHECK PASSED (not a standing-skill pass; no training)', flush=True)
except BaseException as error:
    report['interface_passed'] = False
    report['error'] = repr(error)
    report['traceback'] = traceback.format_exc()
    traceback.print_exc()
    raise
finally:
    try:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        target = HERE / ('interface_report_' + stamp + '.json')
        with target.open('x') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print('Stand V1 interface report:', target, flush=True)
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if app is not None:
                app.close()
