"""Bounded original single-task policy trace; inference only, original formal gains.

Matches eval.py's observation/predictor call order. Records before automatic reset.
No exports, edits to existing files, optimizer updates, or gain changes.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import inspect
from pathlib import Path
import sys
import traceback

ROOT = Path('/workspace/projects/PACE-ICRA2026')
CHECKPOINT = ROOT / 'logs/h1_2_table_tennis/2026-10-15_03-34-22_h1_2_pred_4096_full/model_9999.pt'
SOURCE_HASHES = {
    'legged_lab/scripts/eval.py': '2735e8792535a687c4318393c1f9566c184d776dd188a824c8f07a8fd38e2dfa',
    'legged_lab/envs/h1_2_tt/h1_2_tt_env.py': 'bafa249748acc992832a0fa0a76ff80ab58ec7adbc92b75d871ec039eb9d9d02',
    'legged_lab/envs/base/tt_env.py': '34f92afd1c565d386d10f9bf2c0f64fa215aa8679edf9e2e3d373021d864c728',
    'legged_lab/envs/h1_2_tt/h1_2_tt_config.py': '4eacb07ae5b2f4abaf9b0f686b99c71d13bb18e172af92069025cab853cc5329',
}
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=float, default=12.)
parser.add_argument('--episodes', type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
assert 0 < args.seconds <= 12 and 1 <= args.episodes <= 3
assert CHECKPOINT.is_file() and CHECKPOINT.stat().st_size == 14632418, 'CHECKPOINT MISSING OR SIZE CHANGED'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = json.loads((ROOT / 'tools/h1_2/bilateral_install_report.json').read_text())
protected = {**manifest['protected_sha256'], **manifest['new_files_sha256']}
for rel, expected in SOURCE_HASHES.items():
    assert sha(ROOT / rel) == expected, 'CURRENT SOURCE CHANGED: ' + rel
for rel, expected in protected.items():
    assert sha(ROOT / rel) == expected, 'PROTECTED SOURCE CHANGED: ' + rel
checkpoint_sha = sha(CHECKPOINT)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
out_dir = ROOT / 'tools/h1_2' / ('original_policy_trace_' + stamp)
out_dir.mkdir(exist_ok=False)
report = {
    'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': checkpoint_sha,
    'task': 'h1_2_tt_eval', 'requested_seconds': args.seconds, 'maximum_episodes': args.episodes,
    'learning_updates': 0, 'gains_pose_targets_not_overridden': True,
    'scope': 'bounded original eval-condition policy trace; not a controlled comparison to clean neutral-stance fixtures or formal hit-rate evaluation',
    'predictor_call_order': 'policy(obs); env.step(actions); record_ball_positions(); maybe_predict_and_update_env(); retain returned obs, matching eval.py',
    'after_predictor_tensor_scope': 'after env.step, so after automatic reset when reset is true; never confuse with pre-reset task tensors',
    'samples': [], 'episodes': [], 'predictor_updates_called': 0,
}
app = None
env = None
recording = False
context = {}

try:
    app = AppLauncher(args).app
    import torch
    import isaaclab.utils.math as math_utils
    import legged_lab.envs
    from legged_lab.utils import task_registry
    from legged_lab.assets.unitree.h1_2_tt import H1_2_CONTROL_JOINTS
    from rsl_rl.rsl_rl.runners import OnPolicyPredictorRegressionRunner

    def vec(t):
        assert bool(torch.isfinite(t).all()), 'NONFINITE TELEMETRY'
        return t.detach().cpu().tolist()

    def tensor_value(x):
        return vec(torch.as_tensor(x))

    cfg, agent_cfg = task_registry.get_cfgs('h1_2_tt_eval')
    cfg, agent_cfg = deepcopy(cfg), deepcopy(agent_cfg)
    # Match standard eval.py, overriding only the requested one-env trace size/seed/device.
    cfg.noise.add_noise = True
    cfg.domain_rand.events.push_robot = None
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 5
    cfg.scene.height_scanner.drift_range = (0., 0.)
    if cfg.scene.terrain_generator is not None:
        cfg.scene.terrain_generator.num_rows = 5
        cfg.scene.terrain_generator.num_cols = 5
        cfg.scene.terrain_generator.curriculum = False
        cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)
    agent_cfg.seed = 42
    cfg.scene.seed = agent_cfg.seed
    cfg.device = args.device
    agent_cfg.device = args.device
    Parent = task_registry.get_task_class('h1_2_tt_eval')

    class TraceEnv(Parent):
        def check_reset(self):
            reset, timeout = super().check_reset()
            if recording:
                report['samples'].append(capture(self, reset, timeout))
            return reset, timeout

    def task_tensors(e):
        names = set(('paddle_pos', 'ball_pos', 'ball_vel', 'robot_pos', 'robot_linvel',
                     'ball_future_pose', 'robot_future_pos', 'robot_future_vel',
                     'ball_contact_rew', 'has_touch_paddle', 'has_touch_opponent_table_just_now'))
        names.update(n for n in vars(e) if 'pred' in n or 'future' in n)
        result = {}
        for name in sorted(names):
            value = getattr(e, name, None)
            if isinstance(value, torch.Tensor) and value.numel() <= 64:
                result[name] = vec(value)
        return result

    def capture(e, reset, timeout):
        d = e.robot.data
        q, v = d.joint_pos[0], d.joint_vel[0]
        target = d.joint_pos_target[0]
        pd = kp * (target - q) + kd * (d.joint_vel_target[0] - v) + d.joint_effort_target[0]
        com = (d.body_com_pos_w[0] * masses[:, None]).sum(0) / masses.sum()
        feet = d.body_link_pos_w[0, foot_ids]
        raw_pitch = math_utils.euler_xyz_from_quat(d.root_link_quat_w)[1][0]
        pitch = torch.remainder(raw_pitch + torch.pi, 2 * torch.pi) - torch.pi
        return {
            'global_step': context['step'], 'episode': context['episode'],
            'time_s': float(e.episode_length_buf[0].item()) * e.step_dt,
            'base_z': float((d.root_link_pos_w[0, 2] - e.scene.env_origins[0, 2]).item()),
            'root_pos_env': vec(d.root_link_pos_w[0] - e.scene.env_origins[0]),
            'root_quat_w': vec(d.root_link_quat_w[0]),
            'pitch_rad': float(pitch),
            'tilt_rad': float(torch.acos((-d.projected_gravity_b[0, 2]).clamp(-1, 1))),
            'raw_policy_actions': vec(context['actions'][0]),
            'policy_input_observation': vec(context['obs'][0]),
            'q': vec(q), 'qd': vec(v), 'q_target': vec(target),
            'target_minus_default': vec(target - d.default_joint_pos[0]),
            'q_minus_target': vec(q - target),
            'pd_plus_ff_estimate_Nm': vec(pd),
            'pd_estimate_abs_over_effort_limit': vec(pd.abs() / effort_limits),
            'whole_body_com_env': vec(com - e.scene.env_origins[0]),
            'whole_body_com_minus_foot_link_mean_w': vec(com - feet.mean(0)),
            'feet_pos_env': vec(feet - e.scene.env_origins[0]),
            'feet_quat_w': vec(d.body_link_quat_w[0, foot_ids]),
            'feet_forces_w': vec(e.contact_sensor.data.net_forces_w[0, e.feet_cfg.body_ids]),
            'all_sensor_body_net_contact_forces_w': vec(e.contact_sensor.data.net_forces_w[0]),
            'task_tensors_pre_reset': task_tensors(e),
            'reset': bool(reset[0]), 'timeout': bool(timeout[0]),
        }

    env = TraceEnv(cfg, args.headless)
    assert not env.robot.is_fixed_base and not cfg.scene.robot.spawn.rigid_props.disable_gravity
    view = env.robot.root_physx_view
    kp = view.get_dof_stiffnesses()[0].clone().to(env.device)
    kd = view.get_dof_dampings()[0].clone().to(env.device)
    effort_limits = view.get_dof_max_forces()[0].clone().to(env.device)
    masses = view.get_masses()[0].clone().to(env.device)
    joint_names = list(env.robot.joint_names)
    action_ids = torch.as_tensor(env.action_joint_ids).cpu().tolist()
    action_names = [joint_names[j] for j in action_ids]
    assert action_names == list(H1_2_CONTROL_JOINTS), 'ACTION ORDER MISMATCH'
    assert len(joint_names) == len(action_names) == 27
    foot_ids, foot_names = env.robot.find_bodies(['left_ankle_roll_link', 'right_ankle_roll_link'], preserve_order=True)
    lower_names = [f'{s}_{p}_joint' for s in ('left', 'right') for p in ('hip_pitch', 'knee', 'ankle_pitch')]
    lower_ids = [joint_names.index(n) for n in lower_names]
    assert vec(kp[lower_ids]) == [200., 200., 60., 200., 200., 60.], 'EXPECTED ORIGINAL FORMAL GAINS'
    assert bool((effort_limits > 0).all()) and masses.sum() > 0
    report.update({
        'joint_names': joint_names, 'action_joint_names': action_names, 'action_joint_ids': action_ids,
        'runtime_kp': vec(kp), 'runtime_kd': vec(kd), 'runtime_effort_limits': vec(effort_limits),
        'default_joint_pos': vec(env.robot.data.default_joint_pos[0]),
        'action_scale': tensor_value(env.action_scale), 'clip_actions': tensor_value(env.clip_actions),
        'action_delay_enabled': bool(cfg.domain_rand.action_delay.enable),
        'perception_delay_enabled': bool(cfg.domain_rand.perception_delay.enable),
        'observation_noise_enabled': bool(cfg.noise.add_noise), 'seed': 42,
        'step_dt': env.step_dt, 'mass_kg': float(masses.sum()), 'foot_names': foot_names,
        'contact_sensor_body_names': list(getattr(env.contact_sensor, 'body_names', [])),
        'reset_check_source': inspect.getsource(Parent.check_reset),
        'empirical_normalization_config': getattr(agent_cfg, 'empirical_normalization', None),
        'torque_semantics': 'same-state PD estimates, not measured PhysX drive force',
    })
    runner = OnPolicyPredictorRegressionRunner(env, agent_cfg.to_dict(), log_dir=str(out_dir / 'runner'), device=agent_cfg.device)
    saved_agent_yaml = CHECKPOINT.parent / 'params/agent.yaml'
    if saved_agent_yaml.is_file():
        report['saved_agent_yaml'] = saved_agent_yaml.read_text()
    runner.load(str(CHECKPOINT), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    runner.alg.policy.eval()
    runner._predictor.eval()  # eval.py also does this while exporting; no export here.
    normalizer = getattr(runner, 'obs_normalizer', None)
    if isinstance(normalizer, torch.nn.Module):
        normalizer.eval()
    assert callable(runner._record_ball_positions) and callable(runner._maybe_predict_and_update_env)

    # Verify the loaded policy, predictor and (when applicable) normalizer against checkpoint tensors.
    saved = torch.load(str(CHECKPOINT), map_location='cpu', weights_only=False)
    def dictionaries(value, path='checkpoint', depth=0):
        if isinstance(value, dict):
            yield path, value
            if depth < 3:
                for key, child in value.items():
                    if isinstance(child, dict):
                        yield from dictionaries(child, path + '.' + str(key), depth + 1)
    def cpu_state(module):
        state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        assert all(bool(torch.isfinite(v).all()) for v in state.values()), 'NONFINITE LOADED STATE'
        return state
    modules = {'policy': runner.alg.policy, 'predictor': runner._predictor}
    if isinstance(normalizer, torch.nn.Module):
        modules['obs_normalizer'] = normalizer
    states = {name: cpu_state(module) for name, module in modules.items()}
    matches = {}
    for name, state in states.items():
        if not state:
            matches[name] = 'empty state (no learned normalization state to check)'
            continue
        candidates = []
        for path, mapping in dictionaries(saved):
            if all(k in mapping and isinstance(mapping[k], torch.Tensor)
                   and torch.equal(v, mapping[k].detach().cpu()) for k, v in state.items()):
                candidates.append(path)
        assert candidates, 'LOADED STATE DOES NOT EXACTLY MATCH CHECKPOINT: ' + name
        matches[name] = candidates
    report['checkpoint_state_matches'] = matches
    del saved
    print('[POLICY] CHECKPOINT:', CHECKPOINT, flush=True)
    print('[POLICY] LOADED_STATE_MATCHES:', matches, flush=True)
    print('[POLICY] original Kp/Kd/effort limits:', [(n, float(kp[j]), float(kd[j]), float(effort_limits[j])) for n, j in zip(lower_names, lower_ids)], flush=True)
    print('[POLICY] action_scale / clip / delay / noise:', report['action_scale'], report['clip_actions'], report['action_delay_enabled'], report['observation_noise_enabled'], flush=True)
    with torch.inference_mode():
        obs, _ = env.get_observations()
        assert obs.shape == (1, 495), 'UNEXPECTED POLICY OBSERVATION SHAPE'
        context['episode'] = 0
        recording = True
        for step in range(max(1, round(args.seconds / env.step_dt))):
            assert bool(torch.isfinite(obs).all()), 'NONFINITE POLICY OBSERVATION'
            actions = policy(obs)
            assert actions.shape == (1, 27) and bool(torch.isfinite(actions).all()), 'INVALID POLICY ACTION'
            context.update({'step': step, 'actions': actions.clone(), 'obs': obs.clone()})
            obs, reward, done, extras = env.step(actions)
            assert bool(torch.isfinite(reward).all()) and bool(torch.isfinite(obs).all())
            # Deliberately do not suppress errors as the standard eval loop does.
            runner._record_ball_positions()
            runner._maybe_predict_and_update_env()
            report['predictor_updates_called'] += 1
            row = report['samples'][-1]
            row['task_tensors_after_predictor'] = task_tensors(env)
            if row['reset']:
                report['episodes'].append({'episode': context['episode'], 'duration_s': row['time_s'], 'timeout': row['timeout']})
                context['episode'] += 1
                if len(report['episodes']) >= args.episodes:
                    break
        recording = False
    for name, module in modules.items():
        after = cpu_state(module)
        assert after.keys() == states[name].keys() and all(torch.equal(after[k], v) for k, v in states[name].items()), 'MODEL STATE CHANGED DURING INFERENCE: ' + name
    assert sha(CHECKPOINT) == checkpoint_sha, 'CHECKPOINT FILE CHANGED'
    for rel, expected in {**protected, **SOURCE_HASHES}.items():
        assert sha(ROOT / rel) == expected, 'SOURCE CHANGED DURING TRACE: ' + rel
    report['inference_model_states_unchanged'] = True
    report['source_and_checkpoint_files_unchanged'] = True
    report['completed'] = True

    for ep in sorted({r['episode'] for r in report['samples']}):
        rows = [r for r in report['samples'] if r['episode'] == ep]
        first = next((i for i, r in enumerate(rows) if r['tilt_rad'] > .25), None)
        print('[POLICY] EPISODE', ep, 'duration_s=', round(rows[-1]['time_s'], 3),
              'first_tilt_s=', None if first is None else round(rows[first]['time_s'], 3),
              'reset=', rows[-1]['reset'], 'timeout=', rows[-1]['timeout'],
              'min_base_z=', round(min(r['base_z'] for r in rows), 4), flush=True)
        chosen = {0, len(rows) - 1}
        for t in (.2, .8, 1., 2.):
            if t <= rows[-1]['time_s']:
                chosen.add(min(range(len(rows)), key=lambda i: abs(rows[i]['time_s'] - t)))
        if first is not None:
            chosen.add(first)
        knees = [joint_names.index(s + '_knee_joint') for s in ('left', 'right')]
        for i in sorted(chosen):
            r = rows[i]
            print('[POLICY]', ep, 't=', round(r['time_s'], 3), 'pitch=', round(r['pitch_rad'], 4),
                  'COMfootX=', round(r['whole_body_com_minus_foot_link_mean_w'][0], 4),
                  'knee_target=', [round(r['q_target'][j], 4) for j in knees],
                  'knee_actual=', [round(r['q'][j], 4) for j in knees], flush=True)
        r = rows[first] if first is not None else rows[-1]
        print('[POLICY] DETAIL episode/t:', ep, r['time_s'], 'joint raw_action target_offset target actual error PD_est/limit', flush=True)
        for n, j in zip(lower_names, lower_ids):
            a = action_names.index(n)
            vals = [r['raw_policy_actions'][a], r['target_minus_default'][j], r['q_target'][j], r['q'][j], r['q_minus_target'][j], r['pd_estimate_abs_over_effort_limit'][j]]
            print(n, *[round(v, 5) for v in vals], flush=True)
        selected = {k: v for k, v in r['task_tensors_pre_reset'].items()
                    if k in ('robot_future_pos', 'robot_future_vel', 'robot_pos', 'robot_linvel', 'ball_future_pose', 'paddle_pos')}
        print('[POLICY] TASK_TARGETS_PRE_RESET:', selected, flush=True)
    print('[POLICY] predictor calls:', report['predictor_updates_called'], '; model states unchanged:', report['inference_model_states_unchanged'], flush=True)
except BaseException as error:
    recording = False
    report['completed'] = False
    report['error'] = repr(error)
    report['traceback'] = traceback.format_exc()
    traceback.print_exc()
    raise
finally:
    try:
        with (out_dir / 'report.json').open('x') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print('Policy trace report:', out_dir / 'report.json', flush=True)
        if report.get('completed'):
            print('H1-2 ORIGINAL POLICY TRACE FINISHED (inference only; no training or formal changes)', flush=True)
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if app is not None:
                app.close()
