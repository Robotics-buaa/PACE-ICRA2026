"""Container entry: diagnose, accept, train or evaluate; explicit training budget."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import traceback

from paths import ROOT, HERE, RUN, STAND_SHA, EXPERIMENT, verify, write_json, new_output, sha
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=('diagnose', 'accept', 'train', 'evaluate'))
parser.add_argument('--seconds', type=float, default=1., help='diagnose only: 0.02 to 2 seconds; cannot authorize training')
parser.add_argument('--num-envs', type=int, default=8)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--iterations', type=int)
parser.add_argument('--accept-report', type=Path)
parser.add_argument('--checkpoint', type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.mode == 'diagnose' and not .02 <= args.seconds <= 2.:
    parser.error('diagnose seconds must be in .02..2; acceptance remains fixed at 20s')
if args.mode == 'train' and (not args.iterations or not args.accept_report):
    parser.error('train requires --iterations (explicit budget) and --accept-report (passed new reset acceptance).')
if args.mode == 'train' and args.iterations <= 0:
    parser.error('iterations must be positive')
if not 1 <= args.num_envs <= 4096:
    parser.error('num-envs must be in 1..4096')
if args.mode == 'accept' and args.num_envs < 4:
    parser.error('parallel reset acceptance requires at least four environments')
if args.mode == 'evaluate' and not args.checkpoint:
    parser.error('evaluate requires a new 7-action checkpoint')
os.chdir(ROOT)
package = verify()
manifest_sha = sha(HERE/'manifest.json')
if args.mode == 'train':
    args.accept_report.resolve().relative_to(EXPERIMENT.resolve())
    accepted = json.loads(args.accept_report.read_text())
    assert accepted['mode'] == 'accept' and accepted['passed'] is True
    assert accepted['manifest_sha256'] == manifest_sha
    assert accepted['standing_sha256'] == STAND_SHA
    assert accepted['num_envs'] >= 4 and accepted['optimizer_steps'] == 0
    assert accepted['duration_s'] >= 20.-1e-5
output = new_output(args.mode)
report = {'completed': False, 'passed': False, 'mode': args.mode, 'num_envs': args.num_envs,
          'seed': args.seed, 'manifest_sha256': manifest_sha, 'standing_sha256': STAND_SHA,
          'optimizer_steps': 0, 'scope': 'independent ready-reset right-arm task; no old TT policy or predictor',
          'pose_visual_confirmation': 'must be assessed by user; not inferred from numeric acceptance',
          'contact_semantics': 'per-substep filtered PhysX normal force, >.01N, never distance-score success',
          'collision_scope': '22 arm/paddle vs torso/pelvis pairs; not full-body obstacle avoidance',
          'observation_scope': 'new policy sees simulator ball position and velocity; no real-world estimator claim',
          'speed_scope': 'initial conservative .6rad/s shoulder/elbow and .9rad/s wrist curriculum; not final hitting range',
          'long_training_started_by_local_agent': False}
write_json(output/'report.json', report)
app = env = runner = None
try:
    app = AppLauncher(args).app
    import numpy as np
    import random
    import torch
    import legged_lab.envs
    from rsl_rl.runners import OnPolicyRunner
    from task import make_cfg, ReadyTask, FRAME, HISTORY
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    hold_mode = args.mode in ('accept', 'diagnose')
    cfg, fk, audit = make_cfg(args.num_envs, args.device, args.seed, acceptance=hold_mode)
    env = ReadyTask(cfg, fk, audit, args.headless, acceptance=hold_mode)
    report.update(step_dt=env.step_dt, physics_dt=env.e.physics_dt, observations=FRAME*HISTORY, actions=7,
                  source_geometry=audit, joint_names_physx=list(env.e.robot.joint_names),
                  reference_height_m=float(cfg.reference_base_height),
                  ball_config={k: getattr(cfg.ball, k) for k in
                  ('ball_pos_y_range', 'ball_speed_x_range', 'ball_speed_y_range', 'ball_speed_z_range')})
    if not args.headless:
        env.e.sim.set_camera_view((-4.5, -3.2, 2.15), (-1., 0., .95))
    if hold_mode:
        # New reset semantics only, not a repeat of the old default-standing eval.
        all_ids = torch.arange(env.num_envs, device=env.device)
        env.reset(all_ids)
        plant = env.e
        def reset_checks(ids):
            d = plant.robot.data
            assert torch.allclose(d.joint_pos[ids], plant.ready_q[ids], atol=1e-6)
            assert torch.allclose(d.joint_pos_target[ids], plant.ready_q[ids], atol=1e-6)
            assert (d.joint_vel[ids].abs() < 1e-6).all()
            assert (d.joint_vel_target[ids].abs() < 1e-6).all()
            assert (env.last_action[ids] == 0).all()
            assert torch.allclose(plant.applied27[ids], plant._ready_residual[ids], atol=1e-6)
            assert torch.allclose(env.stand_history[ids, :, 60:87],
                                  (plant._ready_residual[ids]*plant.obs_scales.actions)[:, None, :], atol=1e-6)
            assert torch.allclose(env.history[ids], env.history[ids, :1].expand(-1, HISTORY, -1), atol=1e-6)
            assert torch.allclose(env.stand_history[ids], env.stand_history[ids, :1].expand(-1, 5, -1), atol=1e-6)
            assert (env.actual_margins()[ids] >= env.hard).all()
        reset_checks(all_ids)
        zero = torch.zeros((env.num_envs, 7), device=env.device)
        target, command = env.select_targets(zero)
        assert torch.allclose(target[:, env.right], plant.ready_q[:, env.right], atol=1e-6), 'ZERO ACTION TARGET IS NOT READY'
        assert not env.no_path.any(), 'INITIAL ZERO ACTION REJECTED'
        report['reset_state_and_zero_target_passed'] = True
        duration = 20. if args.mode == 'accept' else args.seconds
        steps = round(duration/env.step_dt)
        low_h = torch.full((env.num_envs,), 1e6, device=env.device)
        high_tilt = torch.zeros_like(low_h)
        high_drift = torch.zeros_like(low_h)
        peak_contact = torch.zeros_like(low_h)
        low_clearance = torch.full((env.num_envs, 3), 1e6, device=env.device)
        high_load = torch.zeros_like(low_h)
        support = torch.zeros_like(low_h)
        arm_error = torch.zeros_like(low_h)
        trace = []
        started = time.perf_counter()
        print('[READY-'+args.mode.upper()+']', args.num_envs, 'envs;', steps*env.step_dt,
              'seconds; direct ready resets, zero right-arm actions; frozen stance feedback.', flush=True)
        with torch.no_grad():
            for step in range(steps):
                if not args.headless and not app.is_running():
                    raise RuntimeError('GUI CLOSED BEFORE ACCEPTANCE COMPLETED')
                obs, reward, done, extras = env.step(zero)
                row = env.last_evidence
                low_h = torch.minimum(low_h, row['height'])
                high_tilt = torch.maximum(high_tilt, row['tilt'])
                high_drift = torch.maximum(high_drift, row['drift'])
                peak_contact = torch.maximum(peak_contact, row['body_peak'])
                low_clearance = torch.minimum(low_clearance, row['min_margins'])
                high_load = torch.maximum(high_load, row['load'])
                support += (row['feet'] > 5.).all(-1).float()
                arm_error = torch.maximum(arm_error, (row['actual_q'][:, env.right]-plant.ready_q[:, env.right]).abs().amax(-1))
                if args.mode == 'diagnose' or step % 10 == 0 or done.any():
                    trace.append({k: v.detach().cpu().tolist() for k, v in row.items()})
                report['duration_s'] = (step+1)*env.step_dt
                assert not row['guard_failure'].any(), 'READY HOLD FAILED: guard/body criterion; see trace'
                assert not done.any() or step == steps-1 and row['timeout'].all(), 'EARLY EPISODE END'
                assert torch.allclose(row['target_q'][:, env.right], plant.ready_q[:, env.right], atol=1e-6), 'ZERO-ACTION TARGET DRIFT'
                if not args.headless:
                    remaining = started+(step+1)*env.step_dt-time.perf_counter()
                    if remaining > 0:
                        time.sleep(min(remaining, env.step_dt))
        report['duration_s'] = steps*env.step_dt
        report['hold_metrics'] = dict(min_height_m=low_h.cpu().tolist(), max_tilt_rad=high_tilt.cpu().tolist(),
                                      max_drift_m=high_drift.cpu().tolist(), peak_body_contact_N=peak_contact.cpu().tolist(),
                                      min_margins_m=low_clearance.cpu().tolist(), max_estimated_load_ratio=high_load.cpu().tolist(),
                                      both_feet_fraction=(support/steps).cpu().tolist(), max_right_tracking_error_rad=arm_error.cpu().tolist())
        print('[READY-HOLD] METRICS:', json.dumps(report['hold_metrics']), flush=True)
        last = env.last_evidence
        print('[READY-HOLD] RIGHT_ARM_ENV0:', json.dumps([
            dict(joint=plant.robot.joint_names[j], actual_rad=float(last['actual_q'][0, j]),
                 target_rad=float(last['target_q'][0, j]), ready_rad=float(plant.ready_q[0, j]),
                 velocity_radps=float(last['velocity'][0, j])) for j in env.right]), flush=True)
        if args.mode == 'accept':
            # Initial engineering criteria, not full return/robustness acceptance.
            assert (low_h >= float(cfg.reference_base_height)-.08).all()
            assert (high_tilt <= .15).all() and (high_drift <= .10).all()
            assert (support/steps >= .9).all() and (high_load <= 1.2).all()
            assert (arm_error <= .08).all(), 'READY PD TRACKING ERROR EXCEEDS .08rad'
            # Dirty selected rows; verify sparse resets clear those histories and flags
            # while retaining all other rows exactly. No physics step is performed here.
            ids = all_ids[::2]
            other = all_ids[1::2]
            kept = {k: x[other].clone() for k, x in {'q': plant.robot.data.joint_pos,
                     'target': plant.robot.data.joint_pos_target, 'history': env.history,
                     'standing': env.stand_history, 'action': env.last_action,
                     'ball': plant.ball.data.root_state_w}.items()}
            env.history[ids] = 7.
            env.stand_history[ids] = 8.
            env.last_action[ids] = .5
            dirty_q = plant.ready_q[ids].clone()
            dirty_q[:, env.right] += .01
            plant.robot.write_joint_state_to_sim(dirty_q, torch.full_like(dirty_q, .1), env_ids=ids)
            for f in env.flags.values():
                f[ids] = True
            env.reset(ids)
            reset_checks(ids)
            for f in env.flags.values():
                assert not f[ids].any()
            for k, x in {'q': plant.robot.data.joint_pos, 'target': plant.robot.data.joint_pos_target,
                         'history': env.history, 'standing': env.stand_history, 'action': env.last_action,
                         'ball': plant.ball.data.root_state_w}.items():
                assert torch.equal(x[other], kept[k]), 'SPARSE RESET CONTAMINATED OTHER ENVS: '+k
            report['parallel_reset_isolation_passed'] = True
            assert env.feedback_calls >= steps
            report['frozen_feedback_calls'] = env.feedback_calls
            # Construct only; acceptance never steps the optimizer. Verify learned
            # policy output really has 7 dimensions and can be handed to this task.
            agent = torch.load(str(RUN/'runner_config.pt'), map_location='cpu', weights_only=False)
            agent = deepcopy(agent)
            agent['algorithm']['class_name'] = 'PPO'
            agent['policy']['class_name'] = 'ActorCritic'
            agent['resume'] = False
            runner = OnPolicyRunner(env, deepcopy(agent), log_dir=None, device=args.device)
            policy = runner.get_inference_policy(device=args.device)
            with torch.no_grad():
                test_actions = policy(env.get_observations()[0])
            assert test_actions.shape == (env.num_envs, 7) and torch.isfinite(test_actions).all()
            assert not runner.alg.optimizer.state, 'UNEXPECTED ACCEPTANCE OPTIMIZER STATE'
            report['runner_7_action_interface_passed'] = True
            report['verdict'] = 'READY_RESET_SHORT_ACCEPTANCE_PASS; hitting ability untested'
        else:
            report['verdict'] = 'ZERO_ACTION_DIAGNOSTIC_COMPLETED; not acceptance, no training authorization'
            report['tracking_within_acceptance_limit'] = bool((arm_error <= .08).all())
            report['frozen_feedback_calls'] = env.feedback_calls
        write_json(output/'trace.json', trace)
    else:
        agent = torch.load(str(RUN/'runner_config.pt'), map_location='cpu', weights_only=False)
        agent = deepcopy(agent)
        agent['algorithm']['class_name'] = 'PPO'
        agent['policy']['class_name'] = 'ActorCritic'
        agent['policy']['init_noise_std'] = .2
        agent['resume'], agent['logger'] = False, 'tensorboard'
        agent['experiment_name'] = 'h1_2_hitting_ready_v1'
        agent['save_interval'] = 50
        agent['num_steps_per_env'] = 24
        torch.save(deepcopy(agent), str(output/'runner_config.pt'))
        runner = OnPolicyRunner(env, deepcopy(agent), log_dir=str(output), device=args.device)
        if args.mode == 'train':
            report.update(status='training', iterations=args.iterations,
                          control_transitions=args.num_envs*24*args.iterations,
                          acceptance_report=str(args.accept_report), resume=False)
            write_json(output/'report.json', report)
            print('[READY-TRAIN] New 7-action PPO; frozen stand checkpoint; no TT load. Explicit budget:', report['control_transitions'], 'transitions.', flush=True)
            before = {k: v.detach().cpu().clone() for k, v in runner.alg.policy.state_dict().items()}
            runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
            optimizer_steps = [float(v['step']) for v in runner.alg.optimizer.state.values() if 'step' in v]
            assert optimizer_steps and max(optimizer_steps) > 0
            assert any(not torch.equal(before[k], v.cpu()) for k, v in runner.alg.policy.state_dict().items())
            assert all(torch.isfinite(v).all() for v in runner.alg.policy.state_dict().values())
            report['optimizer_steps'] = max(optimizer_steps)
            # Explicit final save: no assumptions about the runner's final index.
            runner.save(str(output/'model_final.pt'))
            report.update(checkpoint=str(output/'model_final.pt'), checkpoint_sha256=sha(output/'model_final.pt'),
                          verdict='TRAINING_COMPLETED; physical hitting acceptance pending')
        else:
            cp = args.checkpoint.resolve()
            cp.relative_to(EXPERIMENT.resolve())
            prior = json.loads((cp.parent/'report.json').read_text())
            assert prior['mode'] == 'train' and prior['completed'] and prior['manifest_sha256'] == manifest_sha
            assert sha(cp) == prior['checkpoint_sha256']
            runner.load(str(cp), load_optimizer=False)
            policy = runner.get_inference_policy(device=env.device)
            evaluation_trace = []
            with torch.no_grad():
                for step in range(round(20./env.step_dt)):
                    _, _, done, _ = env.step(policy(env.get_observations()[0]))
                    if step % 5 == 0 or done.any() or (env.last_evidence['paddle_peak'] > .01).any():
                        evaluation_trace.append({'evaluation_step': step, **{k: v.detach().cpu().tolist() for k, v in env.last_evidence.items()}})
            write_json(output/'evaluation_trace.json', evaluation_trace)
            report.update(checkpoint=str(cp), totals=env.total,
                          verdict='EVALUATION_RECORDED; inspect failure/contact/return counts; no automatic skill pass')
    env.verify_frozen()
    verify()
    report.update(completed=True, passed=args.mode == 'accept', frozen_standing_unchanged=True,
                  original_files_unchanged=True, totals=env.total)
except BaseException as error:
    report.update(error=repr(error), traceback=traceback.format_exc())
    if env is not None and hasattr(env, 'last_evidence'):
        write_json(output/'failure_last_transition.json', {k: v.detach().cpu().tolist() for k, v in env.last_evidence.items()})
        last = env.last_evidence
        report['last_transition_summary'] = {k: last[k].detach().cpu().tolist() for k in
            ('episode_step', 'guard_failure', 'no_path', 'height', 'tilt', 'drift', 'body_peak', 'min_margins')}
        print('[READY-FAILURE] SUMMARY:', json.dumps(report['last_transition_summary']), flush=True)
        failed = last['guard_failure'].nonzero(as_tuple=False).flatten()
        if len(failed):
            i = int(failed[0])
            detail = {'env': i, **{k: v[i].detach().cpu().tolist() for k, v in last.items() if k.startswith('guard_')}}
            print('[READY-FAILURE] CANDIDATE_DETAIL:', json.dumps(detail), flush=True)
    if 'trace' in locals():
        write_json(output/'trace.json', trace)
    traceback.print_exc()
    raise
finally:
    try:
        if runner is not None and runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        write_json(output/'report.json', report)
        print('[READY-'+args.mode.upper()+'] REPORT:', output/'report.json', flush=True)
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if app is not None:
                app.close()
