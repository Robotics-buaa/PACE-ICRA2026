"""Inference of the exact stage1 checkpoint; collect every metric BEFORE auto reset."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import traceback

from common import (EXPERIMENT, check_runtime, sha, snapshot_originals, verify_originals, write_json)
from metrics import EpisodeMetrics, THRESHOLDS
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description='20s nominal first-episode evaluation of a new standing-stage checkpoint.')
parser.add_argument('--run-dir', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
assert not args.distributed
run_dir = args.run_dir.resolve()
assert run_dir.parent == EXPERIMENT.resolve(), 'EXPECTED A DIRECT CHILD OF THE NEW STANDING EXPERIMENT'
training = __import__('json').loads((run_dir/'stage1_training_report.json').read_text())
assert training['status'] == 'training_completed' and training['final_iteration'] == 499
assert training['num_envs'] == 512 and training['iterations'] == 500 and not training['resume']
checkpoint = run_dir/'model_499.pt'
assert checkpoint.resolve() == Path(training['checkpoint']).resolve()
assert sha(checkpoint) == training['checkpoint_sha256'], 'STANDING CHECKPOINT CHANGED'
originals = snapshot_originals()
evaluation_dir = run_dir/('evaluation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
evaluation_dir.mkdir(exist_ok=False)
report = {'completed': False, 'learning_updates': 0, 'checkpoint': str(checkpoint),
          'checkpoint_sha256': sha(checkpoint), 'num_envs': 8, 'requested_seconds': 20.,
          'scope': 'first episode of eight nominal reset clones; no disturbances or randomization; not robustness validation',
          'thresholds_are_initial_engineering_criteria': True, 'thresholds': THRESHOLDS,
          'torque_semantics': 'same-state implicit PD+FF estimate, not measured PhysX drive force',
          'metrics_scope': 'all control-rate samples from each first episode, before automatic reset; no settling samples omitted'}
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
    cfg, agent = task_registry.get_cfgs('h1_2_stand_v1_eval')
    cfg, agent = deepcopy(cfg), deepcopy(agent)
    cfg.scene.num_envs, cfg.scene.seed, cfg.device = 8, 42, args.device
    agent.device, agent.seed, agent.resume = args.device, 42, False
    Parent = task_registry.get_task_class('h1_2_stand_v1_eval')

    class MeasuredEnv(Parent):
        def check_reset(self):
            reset, timeout = super().check_reset()
            if recording:
                capture(self, reset, timeout)
            return reset, timeout

    def finite_list(tensor):
        assert torch.isfinite(tensor).all(), 'NONFINITE EVALUATION STATE'
        return tensor.detach().cpu().tolist()

    def capture(e, reset, timeout):
        d = e.robot.data
        position = d.root_link_pos_w-e.scene.env_origins
        h = position[:, 2]
        tilt = torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1., 1.))
        drift = torch.linalg.vector_norm(position[:, :2]-e.stand_goal_env()[:, :2], dim=1)
        speed = torch.linalg.vector_norm(d.root_lin_vel_w, dim=1)
        forces = e.contact_sensor.data.net_forces_w[:, e.feet_cfg.body_ids, 2]
        assert forces.shape == (8, 2)
        error = d.joint_pos_target-d.joint_pos
        estimate = (kp*error + kd*(d.joint_vel_target-d.joint_vel) + d.joint_effort_target)
        load = estimate.abs()/limits
        projected = (e.stand_projection_delta.abs() > 1e-6).any(dim=1)
        batch = {name: finite_list(value) for name, value in {
            'time_s': e.episode_length_buf*e.step_dt, 'height_m': h, 'tilt_rad': tilt,
            'xy_drift_m': drift, 'speed_mps': speed, 'foot_fz_N': forces,
            'tracking_error_rad': error, 'estimated_load_ratio': load,
            'projection_applied': projected, 'reset': reset, 'timeout': timeout,
        }.items()}
        for i, metric in enumerate(stats):
            if metric.ended:
                continue
            row = {name: value[i] for name, value in batch.items()}
            metric.add(row)
            if metric.count == 1 or metric.count % 10 == 0 or metric.ended:
                traces.append({'env': i, **row})

    env = MeasuredEnv(cfg, args.headless)
    report['runtime_lower_kp'] = check_runtime(env)
    view = env.robot.root_physx_view
    kp = view.get_dof_stiffnesses().clone().to(env.device)
    kd = view.get_dof_dampings().clone().to(env.device)
    limits = view.get_dof_max_forces().clone().to(env.device)
    assert (limits > 0).all()
    reference = float(cfg.reference_base_height)
    report.update(reference_height_m=reference, step_dt=env.step_dt, joint_names=list(env.robot.joint_names))
    stats = [EpisodeMetrics(env.robot.joint_names, reference, env.step_dt) for _ in range(8)]
    # Read the precise trusted training config saved BEFORE runner mutation;
    # do not rely on YAML handling of tuple/function tags in configclasses.
    assert sha(run_dir/'runner_config.pt') == training['runner_config_sha256'], 'RUNNER CONFIG CHANGED'
    train_cfg = torch.load(str(run_dir/'runner_config.pt'), map_location='cpu', weights_only=False)
    assert train_cfg['algorithm']['class_name'] == 'PPO' and train_cfg['policy']['class_name'] == 'ActorCritic'
    runner = OnPolicyRunner(env, deepcopy(train_cfg), log_dir=None, device=args.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    saved = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    modules = {'policy': runner.alg.policy, 'obs_normalizer': runner.obs_normalizer,
               'privileged_obs_normalizer': runner.privileged_obs_normalizer}
    states = {name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
              for name, module in modules.items()}
    for name, key in (('policy', 'model_state_dict'), ('obs_normalizer', 'obs_norm_state_dict'),
                      ('privileged_obs_normalizer', 'privileged_obs_norm_state_dict')):
        if states[name]:
            assert key in saved and saved[key].keys() == states[name].keys(), 'LOADED STATE KEY MISMATCH: '+name
            assert all(torch.equal(v, saved[key][k].cpu()) for k, v in states[name].items()), 'LOADED STATE MISMATCH: '+name
    del saved
    with torch.inference_mode():
        env.reset(torch.arange(8, device=env.device))
        env.compute_intermediate_values()
        obs, _ = env.get_observations()
        assert obs.shape == (8, 495) and (env.episode_length_buf == 0).all()
        recording = True
        print('[STAND-EVAL] Exact checkpoint:', checkpoint, flush=True)
        print('[STAND-EVAL] Recording each FIRST episode before reset; maximum20s; no learning.', flush=True)
        for _ in range(int(env.max_episode_length)):
            assert torch.isfinite(obs).all()
            action = policy(obs)
            assert action.shape == (8, 27) and torch.isfinite(action).all()
            obs, reward, done, extras = env.step(action)
            assert torch.isfinite(reward).all() and torch.isfinite(obs).all()
            if all(item.ended for item in stats):
                break
        recording = False
    for name, module in modules.items():
        assert all(torch.equal(v.cpu(), states[name][k]) for k, v in module.state_dict().items()), 'INFERENCE CHANGED MODEL STATE: '+name
    assert sha(checkpoint) == training['checkpoint_sha256']
    verify_originals(originals)
    results = [item.result() for item in stats]
    report.update(completed=True, results=results,
                  completed_20s_envs=sum(r['completed_20s'] for r in results),
                  kinematic_pass_envs=sum(r['kinematic_standing_pass'] for r in results),
                  stage1_candidate_pass_envs=sum(r['stage1_candidate_pass'] for r in results),
                  original_files_unchanged=True, inference_model_states_unchanged=True)
    for i, row in enumerate(results):
        print('[STAND-EVAL] env', i, 'duration_s=', round(row['duration_s'], 3),
              'min_z=', round(row['min_height_m'], 4), 'max_tilt=', round(row['max_tilt_rad'], 4),
              'max_drift=', round(row['max_xy_drift_m'], 4), '20s=', row['completed_20s'],
              'kinematic_pass=', row['kinematic_standing_pass'],
              'load_flags=', row['estimated_load_flag_joints'], flush=True)
        lower = {name: data for name, data in row['per_joint'].items()
                 if name.endswith(('_knee_joint', '_ankle_pitch_joint'))}
        print('[STAND-EVAL] knees/ankle_pitch:', lower, flush=True)
    print('[STAND-EVAL] SUMMARY: completed20s / kinematic / candidate =',
          report['completed_20s_envs'], '/', report['kinematic_pass_envs'], '/', report['stage1_candidate_pass_envs'], 'of8', flush=True)
    print('H1-2 STAND STAGE1 EVALUATION FINISHED (nominal conditions; inspect metrics, not training reward alone)', flush=True)
except BaseException as error:
    recording = False
    report.update(completed=False, error=repr(error), traceback=traceback.format_exc())
    report['partial_results'] = [item.result() for item in stats if item.count]
    traceback.print_exc()
    raise
finally:
    try:
        write_json(evaluation_dir/'evaluation_report.json', report)
        write_json(evaluation_dir/'sampled_first_episode_trace.json', {'samples': traces, 'sampling': 'first, every10th, terminal; summary uses ALL frames'})
        print('[STAND-EVAL] REPORT:', evaluation_dir/'evaluation_report.json', flush=True)
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if app is not None:
                app.close()
