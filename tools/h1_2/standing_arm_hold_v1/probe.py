"""Arm-command ablation with frozen policy and unchanged physics/PD; no pushes or training."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path('/workspace/projects/PACE-ICRA2026')
HERE = Path(__file__).resolve().parent
RUN = ROOT/'logs/h1_2_standing_v1/2026-10-16_13-07-50_805816Z_stand_v1_512x500_seed42'
RECOVERY = ROOT/'tools/h1_2/standing_recovery_v1'
sys.path.insert(0, str(ROOT/'tools/h1_2/standing_v1_stage1'))
from common import check_runtime, sha, snapshot_originals, verify_originals, write_json
from metrics import EpisodeMetrics
sys.path.insert(0, str(RECOVERY))
from recovery_metrics import BAND
sys.path.insert(0, str(HERE))
from quiet_metrics import CONDITIONS, QuietMetrics, arm_action_indices
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description='Frozen-policy arm-target ablation; no training and no physical joint locking.')
parser.add_argument('--condition', choices=['all']+list(CONDITIONS), default='all')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
assert not getattr(args, 'distributed', False)
conditions = list(CONDITIONS) if args.condition == 'all' else [args.condition]
n = len(conditions)
manifest = json.loads((HERE/'manifest.json').read_text())

def verify_probe():
    for name, digest in manifest['files_sha256'].items():
        assert sha(HERE/name) == digest, 'ARM PROBE CHANGED: '+name
    for name, digest in manifest['required_files_sha256'].items():
        assert sha(ROOT/name) == digest, 'BASELINE FILE CHANGED: '+name

verify_probe()
originals = snapshot_originals()
training = json.loads((RUN/'stage1_training_report.json').read_text())
checkpoint = RUN/'model_499.pt'
assert training['status'] == 'training_completed' and training['final_iteration'] == 499
assert Path(training['checkpoint']).resolve() == checkpoint.resolve()
assert sha(checkpoint) == training['checkpoint_sha256']
assert sha(RUN/'runner_config.pt') == training['runner_config_sha256']
reference_path = RUN/'recovery_v1_20261017T024933_806736Z/recovery_report.json'
reference = json.loads(reference_path.read_text())
assert reference['completed'] and reference['checkpoint_sha256'] == training['checkpoint_sha256']
assert reference['recovery_band'] == BAND, 'QUIET THRESHOLDS CHANGED'
output = RUN/('arm_hold_v1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
output.mkdir(exist_ok=False)
report = {'completed': False, 'learning_updates': 0, 'checkpoint': str(checkpoint),
          'checkpoint_sha256': sha(checkpoint), 'conditions': conditions,
          'scope': 'arm-target diagnostic; altered closed-loop action history/state distribution; not a trained final fix',
          'quiet_band_unchanged': BAND, 'saved_reference_report': str(reference_path),
          'reference_pre_fraction': [r['baseline_good_fraction'] for r in reference['results']],
          'reference_tail_fraction': [r['final_window_good_fraction'] for r in reference['results']],
          'comparison_scope': 'saved original-policy reference; no new baseline rerun or randomized matched trial',
          'torque_semantics': 'PD+FF estimate, not measured drive torque'}
app = env = None
stats, traces = [], []
recording = False
try:
    app = AppLauncher(args).app
    import torch
    import legged_lab.envs
    from legged_lab.utils import task_registry
    from rsl_rl.runners import OnPolicyRunner
    from standing_v1.task import register_tasks

    register_tasks()
    cfg, _ = task_registry.get_cfgs('h1_2_stand_v1_eval')
    cfg = deepcopy(cfg)
    cfg.scene.num_envs, cfg.scene.seed, cfg.device = n, 42, args.device
    Parent = task_registry.get_task_class('h1_2_stand_v1_eval')

    class ArmHoldEnv(Parent):
        def step(self, actions):
            if not hasattr(self, 'arm_hold_mask'):
                return super().step(actions)  # constructor callbacks, before test masks exist
            self.arm_raw_actions = actions.detach().clone()
            applied = actions.clone()
            applied[self.arm_hold_mask] = 0.
            self.arm_input_actions = applied.detach().clone()
            return super().step(applied)

        def check_reset(self):
            reset, timeout = super().check_reset()
            if recording:
                capture(self, reset, timeout)
            return reset, timeout

    def finite_list(value):
        assert torch.isfinite(value).all(), 'NONFINITE ARM PROBE STATE'
        return value.detach().cpu().tolist()

    def capture(e, reset, timeout):
        d = e.robot.data
        action_targets = d.joint_pos_target[:, e.action_joint_ids]
        defaults = d.default_joint_pos[:, e.action_joint_ids]
        assert torch.allclose(action_targets[e.arm_hold_mask], defaults[e.arm_hold_mask], atol=1e-6), 'ARM TARGET NOT DEFAULT'
        pos = d.root_link_pos_w-e.scene.env_origins
        error = d.joint_pos_target-d.joint_pos
        estimate = kp*error+kd*(d.joint_vel_target-d.joint_vel)+d.joint_effort_target
        feet = e.contact_sensor.data.net_forces_w[:, e.feet_cfg.body_ids]
        assert feet.shape == (n, 2, 3)
        batch = {name: finite_list(value) for name, value in {
            'time_s': e.episode_length_buf*e.step_dt, 'height_m': pos[:, 2],
            'tilt_rad': torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1., 1.)),
            'xy_drift_m': torch.linalg.vector_norm(pos[:, :2]-e.stand_goal_env()[:, :2], dim=1),
            'speed_mps': torch.linalg.vector_norm(d.root_lin_vel_w, dim=1),
            'angular_speed_rps': torch.linalg.vector_norm(d.root_ang_vel_b, dim=1),
            'root_position_env_m': pos, 'root_com_velocity_world': d.root_vel_w,
            'root_angular_velocity_body_rps': d.root_ang_vel_b,
            'joint_actual_rad': d.joint_pos, 'joint_target_rad': d.joint_pos_target,
            'joint_velocity_radps': d.joint_vel, 'tracking_error_rad': error,
            'estimated_load_ratio': estimate.abs()/limits, 'foot_fz_N': feet[:, :, 2],
            'feet_forces_world_N': feet, 'raw_policy_actions': e.arm_raw_actions,
            'input_actions_after_arm_override': e.arm_input_actions,
            'projection_applied': (e.stand_projection_delta.abs() > 1e-6).any(dim=1),
            'reset': reset, 'timeout': timeout,
        }.items()}
        for i, metric in enumerate(stats):
            if not metric.ended:
                row = {name: values[i] for name, values in batch.items()}
                metric.add(row)
                traces.append({'condition': conditions[i], **row})

    env = ArmHoldEnv(cfg, args.headless)
    report['runtime_lower_kp'] = check_runtime(env)
    held = [arm_action_indices(env.action_joint_names, condition) for condition in conditions]
    env.arm_hold_mask = torch.zeros((n, 27), dtype=torch.bool, device=env.device)
    for i, ids in enumerate(held):
        env.arm_hold_mask[i, ids] = True
    report['held_action_names'] = [[env.action_joint_names[j] for j in ids] for ids in held]
    report['joint_names'], report['action_joint_names'] = list(env.robot.joint_names), list(env.action_joint_names)
    view = env.robot.root_physx_view
    kp = view.get_dof_stiffnesses().clone().to(env.device)
    kd = view.get_dof_dampings().clone().to(env.device)
    limits = view.get_dof_max_forces().clone().to(env.device)
    assert (limits > 0).all()
    report['reference_height_m'], report['step_dt'] = float(cfg.reference_base_height), env.step_dt
    stats = [QuietMetrics(EpisodeMetrics(env.robot.joint_names, float(cfg.reference_base_height), env.step_dt),
                          condition, BAND) for condition in conditions]
    train_cfg = torch.load(str(RUN/'runner_config.pt'), map_location='cpu', weights_only=False)
    assert train_cfg['algorithm']['class_name'] == 'PPO'
    runner = OnPolicyRunner(env, deepcopy(train_cfg), log_dir=None, device=args.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    modules = {'policy': runner.alg.policy, 'actor_normalizer': runner.obs_normalizer,
               'critic_normalizer': runner.privileged_obs_normalizer}
    states = {name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
              for name, module in modules.items()}
    saved = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    for name, key in (('policy', 'model_state_dict'), ('actor_normalizer', 'obs_norm_state_dict'),
                      ('critic_normalizer', 'privileged_obs_norm_state_dict')):
        if states[name]:
            assert saved[key].keys() == states[name].keys()
            assert all(torch.equal(v, saved[key][k].cpu()) for k, v in states[name].items())
    del saved
    if not args.headless:
        origin = env.scene.env_origins[0].detach().cpu().tolist()
        env.sim.set_camera_view([origin[i]+v for i, v in enumerate((-4.5, -3.2, 2.15))],
                                [origin[i]+v for i, v in enumerate((-1.8, 0., .95))])
    with torch.inference_mode():
        env.reset(torch.arange(n, device=env.device))
        env.compute_intermediate_values()
        obs, _ = env.get_observations()
        assert obs.shape == (n, 495) and (env.episode_length_buf == 0).all()
        recording = True
        wall_start = time.perf_counter()
        print('[ARM-HOLD] Fixed model499; NO kicks/learning/PD changes; dynamic joints under original PD.', flush=True)
        print('[ARM-HOLD] Hold mapping:', dict(zip(conditions, report['held_action_names'])), flush=True)
        for step in range(int(env.max_episode_length)):
            if not args.headless and not app.is_running():
                raise RuntimeError('GUI CLOSED BEFORE ARM TEST COMPLETED')
            assert torch.isfinite(obs).all()
            actions = policy(obs)
            assert actions.shape == (n, 27) and torch.isfinite(actions).all()
            obs, reward, done, extras = env.step(actions)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            if not args.headless:
                env.sim.render()
                remaining = wall_start+(step+1)*env.step_dt-time.perf_counter()
                if remaining > 0:
                    time.sleep(min(remaining, env.step_dt))
            if all(metric.ended for metric in stats):
                break
        recording = False
    for name, module in modules.items():
        assert all(torch.equal(value.cpu(), states[name][key]) for key, value in module.state_dict().items())
    assert sha(checkpoint) == training['checkpoint_sha256']
    verify_originals(originals)
    verify_probe()
    results = [metric.result() for metric in stats]
    report.update(completed=True, results=results, original_files_unchanged=True, inference_model_states_unchanged=True)
    for result in results:
        b, windows = result['standing'], result['windows']
        print('[ARM-HOLD] RESULT:', result['condition'], 'duration=', b['duration_s'],
              'standing_pass=', b['stage1_candidate_pass'], 'quiet_pass=', result['quiet_candidate_pass'],
              'pre_good=', windows['pre5s']['good_fraction'], 'tail_good=', windows['tail']['good_fraction'],
              'max_v=', b['max_speed_mps'], 'max_w=', windows['all']['max_angular_speed_rps'],
              'load_flags=', b['estimated_load_flag_joints'], flush=True)
        print('[ARM-HOLD] TAIL_MISSES:', windows['tail']['miss_counts'], flush=True)
        top = sorted(windows['tail']['joint_velocity_rms_radps'].items(), key=lambda item: -(item[1] or 0.))[:6]
        print('[ARM-HOLD] TAIL_JOINT_SPEED_RMS:', top, flush=True)
    print('[ARM-HOLD] FINISHED. Diagnostic action overrides only; no final controller adopted.', flush=True)
except BaseException as error:
    recording = False
    report.update(completed=False, error=repr(error), traceback=traceback.format_exc(),
                  partial_results=[metric.result() for metric in stats if metric.count])
    traceback.print_exc()
    raise
finally:
    try:
        write_json(output/'arm_hold_report.json', report)
        write_json(output/'full_control_rate_trace.json', {'samples': traces, 'scope': 'ALL first-episode control steps; no physics-substep frequency claim'})
        print('[ARM-HOLD] REPORT:', output/'arm_hold_report.json', flush=True)
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if app is not None:
                app.close()
