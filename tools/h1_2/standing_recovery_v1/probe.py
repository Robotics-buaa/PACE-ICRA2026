"""Frozen standing policy: eight small root-COM velocity increments, no learning."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path('/workspace/projects/PACE-ICRA2026')
HERE = Path(__file__).resolve().parent
RUN = ROOT/'logs/h1_2_standing_v1/2026-10-16_13-07-50_805816Z_stand_v1_512x500_seed42'
sys.path.insert(0, str(ROOT/'tools/h1_2/standing_v1_stage1'))
from common import check_runtime, sha, snapshot_originals, verify_originals, write_json
from metrics import EpisodeMetrics
sys.path.insert(0, str(HERE))
from recovery_metrics import (CASES, BAND, PUSH_TIME_S, BASELINE_SECONDS, HOLD_SECONDS,
                              DEADLINE_SECONDS, TAIL_SECONDS, MIN_GOOD_FRACTION,
                              RecoveryMetrics, select_cases)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description='Fixed-policy velocity-perturbation test; no training or gain changes.')
parser.add_argument('--case', choices=['all']+[case['name'] for case in CASES], default='all')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
assert not getattr(args, 'distributed', False), 'SINGLE GPU PROBE'
cases = select_cases(args.case)
n = len(cases)
assert n in (1, 8)
manifest = json.loads((HERE/'manifest.json').read_text())

def verify_probe():
    for name, digest in manifest['files_sha256'].items():
        assert sha(HERE/name) == digest, 'RECOVERY SCRIPT CHANGED: '+name

verify_probe()
originals = snapshot_originals()
training = json.loads((RUN/'stage1_training_report.json').read_text())
checkpoint = RUN/'model_499.pt'
assert training['status'] == 'training_completed' and training['final_iteration'] == 499
assert Path(training['checkpoint']).resolve() == checkpoint.resolve()
assert sha(checkpoint) == training['checkpoint_sha256']
assert sha(RUN/'runner_config.pt') == training['runner_config_sha256']
baseline_path = RUN/'evaluation_20261016T132644_242054Z/evaluation_report.json'
baseline = json.loads(baseline_path.read_text())
assert baseline['completed'] and baseline['stage1_candidate_pass_envs'] == 8
assert baseline['checkpoint_sha256'] == training['checkpoint_sha256'], 'BASELINE MODEL MISMATCH'
output = RUN/('recovery_v1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
output.mkdir(exist_ok=False)
report = {'completed': False, 'learning_updates': 0, 'checkpoint': str(checkpoint),
          'checkpoint_sha256': sha(checkpoint), 'passed_baseline_report': str(baseline_path), 'cases': cases, 'num_envs': n,
          'scope': 'first episode only; small world-horizontal root-COM velocity increments; not calibrated external forces',
          'push_time_s': PUSH_TIME_S, 'episode_seconds': 20.,
          'engineering_thresholds_not_manufacturer_standard': True,
          'recovery_band': BAND, 'pre_push_window_s': BASELINE_SECONDS,
          'continuous_good_window_s': HOLD_SECONDS, 'completion_deadline_after_push_s': DEADLINE_SECONDS,
          'final_good_window_s': TAIL_SECONDS, 'min_good_fraction': MIN_GOOD_FRACTION,
          'torque_semantics': 'implicit PD+FF estimates, not measured drive force'}
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

    class MeasuredEnv(Parent):
        def check_reset(self):
            reset, timeout = super().check_reset()
            if recording:
                capture(self, reset, timeout)
            return reset, timeout

    def finite_list(value):
        assert torch.isfinite(value).all(), 'NONFINITE PROBE STATE'
        return value.detach().cpu().tolist()

    def capture(e, reset, timeout):
        d = e.robot.data
        pos = d.root_link_pos_w-e.scene.env_origins
        error = d.joint_pos_target-d.joint_pos
        estimate = kp*error+kd*(d.joint_vel_target-d.joint_vel)+d.joint_effort_target
        feet_world = e.contact_sensor.data.net_forces_w[:, e.feet_cfg.body_ids]
        feet = feet_world[:, :, 2]
        assert feet.shape == (n, 2)
        batch = {name: finite_list(value) for name, value in {
            'time_s': e.episode_length_buf*e.step_dt, 'height_m': pos[:, 2],
            'tilt_rad': torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1., 1.)),
            'xy_drift_m': torch.linalg.vector_norm(pos[:, :2]-e.stand_goal_env()[:, :2], dim=1),
            'speed_mps': torch.linalg.vector_norm(d.root_lin_vel_w, dim=1),
            'angular_speed_rps': torch.linalg.vector_norm(d.root_ang_vel_b, dim=1),
            'root_com_velocity_world': d.root_vel_w,
            'root_position_env_m': pos, 'root_angular_velocity_body_rps': d.root_ang_vel_b,
            'joint_actual_rad': d.joint_pos, 'joint_target_rad': d.joint_pos_target,
            'joint_velocity_radps': d.joint_vel, 'feet_forces_world_N': feet_world,
            'foot_fz_N': feet, 'tracking_error_rad': error, 'estimated_load_ratio': estimate.abs()/limits,
            'projection_applied': (e.stand_projection_delta.abs() > 1e-6).any(dim=1),
            'reset': reset, 'timeout': timeout,
        }.items()}
        for i, metric in enumerate(stats):
            if metric.ended:
                continue
            row = {name: values[i] for name, values in batch.items()}
            metric.add(row)
            if metric.count == 1 or metric.count % 5 == 0 or metric.ended:
                traces.append({'case': cases[i]['name'], **row})

    env = MeasuredEnv(cfg, args.headless)
    report['runtime_lower_kp'] = check_runtime(env)
    view = env.robot.root_physx_view
    kp = view.get_dof_stiffnesses().clone().to(env.device)
    kd = view.get_dof_dampings().clone().to(env.device)
    limits = view.get_dof_max_forces().clone().to(env.device)
    assert (limits > 0).all()
    reference = float(cfg.reference_base_height)
    report.update(reference_height_m=reference, step_dt=env.step_dt, joint_names=list(env.robot.joint_names))
    stats = [RecoveryMetrics(EpisodeMetrics(env.robot.joint_names, reference, env.step_dt), case) for case in cases]
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
            assert all(torch.equal(v, saved[key][k].cpu()) for k, v in states[name].items()), 'LOADED STATE MISMATCH: '+name
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
        push_step = round(PUSH_TIME_S/env.step_dt)
        assert abs(push_step*env.step_dt-PUSH_TIME_S) < 1e-6
        deltas = torch.tensor([case['delta_v_world_mps'] for case in cases], device=env.device)
        recording = True
        wall_start = time.perf_counter()
        print('[RECOVERY] Frozen checkpoint:', checkpoint, flush=True)
        print('[RECOVERY] Cases:', [case['name'] for case in cases], '; push at5s; no learning.', flush=True)
        for step in range(int(env.max_episode_length)):
            if not args.headless and not app.is_running():
                raise RuntimeError('GUI CLOSED BEFORE RECOVERY TEST COMPLETED')
            if step == push_step:
                ids = torch.tensor([i for i, metric in enumerate(stats) if not metric.ended],
                                   device=env.device, dtype=torch.long)
                if ids.numel():
                    before = env.robot.data.root_vel_w[ids].clone()
                    target = before.clone()
                    target[:, :3] += deltas[ids]
                    # Same API family as IsaacLab2.1's push_by_setting_velocity;
                    # preserve vertical/angular velocities and add ONLY horizontal delta.
                    env.robot.write_root_velocity_to_sim(target, env_ids=ids)
                    actual = env.robot.root_physx_view.get_root_velocities()[ids].clone()
                    assert torch.allclose(actual, target, atol=1e-5, rtol=0.), 'PHYSX VELOCITY WRITE MISMATCH'
                    for i, old, new in zip(ids.cpu().tolist(), finite_list(before), finite_list(actual)):
                        stats[i].mark_push(step*env.step_dt, old, new)
                        print('[RECOVERY] PUSH:', cases[i]['name'], stats[i].push, flush=True)
            assert torch.isfinite(obs).all()
            action = policy(obs)
            assert action.shape == (n, 27) and torch.isfinite(action).all()
            obs, reward, done, extras = env.step(action)
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
        assert all(torch.equal(v.cpu(), states[name][k]) for k, v in module.state_dict().items()), 'PROBE CHANGED MODEL STATE: '+name
    assert sha(checkpoint) == training['checkpoint_sha256']
    verify_originals(originals)
    verify_probe()
    results = [metric.result() for metric in stats]
    report.update(completed=True, results=results, recovery_pass_cases=sum(r['recovery_pass'] for r in results),
                  actual_push_cases=sum(r['push'] is not None for r in results),
                  original_files_unchanged=True, inference_model_states_unchanged=True)
    for result in results:
        body = result['standing']
        print('[RECOVERY] RESULT:', result['case']['name'], 'pass=', result['recovery_pass'],
              'reason=', result['reason'], 'baseline_good=', result['baseline_good_fraction'],
              'recovery_s=', result['recovery_latency_s'], 'tail_good=', result['final_window_good_fraction'],
              'duration_s=', body['duration_s'], 'min_z=', round(body['min_height_m'], 4),
              'max_tilt=', round(body['max_tilt_rad'], 4), 'max_drift=', round(body['max_xy_drift_m'], 4),
              'load_flags=', body['estimated_load_flag_joints'], flush=True)
    print('[RECOVERY] SUMMARY:', report['recovery_pass_cases'], '/', n,
          '; actual pushes:', report['actual_push_cases'], '; model unchanged; no training.', flush=True)
except BaseException as error:
    recording = False
    report.update(completed=False, error=repr(error), traceback=traceback.format_exc(),
                  partial_results=[metric.result() for metric in stats if metric.count])
    traceback.print_exc()
    raise
finally:
    try:
        write_json(output/'recovery_report.json', report)
        write_json(output/'sampled_trace.json', {'samples': traces, 'sampling': 'first/every5th/terminal; summaries use ALL frames'})
        print('[RECOVERY] REPORT:', output/'recovery_report.json', flush=True)
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if app is not None:
                app.close()
