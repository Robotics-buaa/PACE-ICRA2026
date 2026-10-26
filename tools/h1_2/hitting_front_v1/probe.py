"""Real-ball composite inference: frozen stand20 channels + frozen TT right-arm7.

Not a trained combined policy, not a default-pose replacement, no optimizer steps.
"""
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
TT_CHECKPOINT = ROOT/'logs/h1_2_table_tennis/2026-10-15_03-34-22_h1_2_pred_4096_full/model_9999.pt'
sys.path.insert(0, str(ROOT/'tools/h1_2/standing_v1_stage1'))
from common import check_runtime, sha, snapshot_originals, verify_originals, write_json
from metrics import EpisodeMetrics
sys.path.insert(0, str(HERE))
from logic import arm_blend, contact_windows, right_arm_indices, standing_view
from runtime import configure_self_contacts, bind_guard, apply_guard, verify_urdf
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
assert not getattr(args, 'distributed', False)
manifest = json.loads((HERE/'manifest.json').read_text())


def verify_probe():
    for name, digest in manifest['files_sha256'].items():
        assert sha(HERE/name) == digest, 'NEW PROBE CHANGED: '+name
    for name, digest in manifest['required_files_sha256'].items():
        assert sha(ROOT/name) == digest, 'BASELINE CHANGED: '+name


verify_probe()
originals = snapshot_originals()
training = json.loads((RUN/'stage1_training_report.json').read_text())
checkpoint = RUN/'model_499.pt'
assert training['status'] == 'training_completed' and training['final_iteration'] == 499
assert Path(training['checkpoint']).resolve() == checkpoint.resolve()
assert sha(checkpoint) == training['checkpoint_sha256']
assert sha(RUN/'runner_config.pt') == training['runner_config_sha256']
assert TT_CHECKPOINT.is_file() and TT_CHECKPOINT.stat().st_size == 14632418
tt_sha = sha(TT_CHECKPOINT)
previous = RUN/'arm_hold_v1_20261017T030529_751835Z/arm_hold_report.json'
saved_hold = json.loads(previous.read_text())
assert saved_hold['completed'] and saved_hold['checkpoint_sha256'] == training['checkpoint_sha256']
assert any(r['condition'] == 'right_arm_default' and r['quiet_candidate_pass'] for r in saved_hold['results'])
output = RUN/('hitting_front_v1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
output.mkdir(exist_ok=False)
report = {'completed': False, 'learning_updates': 0, 'checkpoint': str(checkpoint),
          'checkpoint_sha256': sha(checkpoint), 'tt_checkpoint': str(TT_CHECKPOINT),
          'tt_checkpoint_sha256': tt_sha, 'previous_arm_hold_report': str(previous),
          'scope': 'NEW body-front guarded controller and explicit self-contact pairs; stationary-body original eval serves',
          'phases': '0-4s cosine front-ready entry, both arms controlled;4-5s settle;5-5.5s ready-to-TT blend;5.5-20s guarded TT requests',
          'contact_semantics': 'PhysX paddle-to-ball filtered NORMAL force sampled EACH physics substep; threshold .01N',
          'post_contact_scope': '2s height/tilt/drift/support/estimated-load screen; not quietness, return or robustness acceptance',
          'torque_semantics': 'implicit PD+FF estimate, not measured drive torque',
          'predictor_calls': 0, 'predictor_changed_calls': 0,
          'predictor_order': 'TT policy(obs); env.step(mixed); record_ball_positions(); maybe_predict_and_update_env(); retain returned obs'}
app = env = None
original_sim_step = None
recording = False
stats = None
samples, contacts = [], []
context = {'step': 0, 'substep': 0, 'self_contact_peak_N': 0.}
arm_guard = None
self_views = []
self_contacts = []
stop_reasons = []
try:
    app = AppLauncher(args).app
    import torch
    import legged_lab.envs
    from legged_lab.utils import task_registry
    from legged_lab.envs.h1_2_tt.h1_2_tt_env import H1_2TTEnv
    from legged_lab.assets.unitree.h1_2_tt import H1_2_PADDLE_BODY_NAME, H1_2_CONTROL_JOINTS
    from standing_v1.task import H1_2StandV1Env, make_cfg
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.rsl_rl.runners import OnPolicyPredictorRegressionRunner

    cfg = make_cfg()
    tt_cfg, agent_cfg = task_registry.get_cfgs('h1_2_tt_eval')
    agent_cfg = deepcopy(agent_cfg)
    cfg.ball = deepcopy(tt_cfg.ball)  # real original eval serves; no custom easy-ball success claims
    cfg.scene.num_envs, cfg.scene.seed, cfg.device = 1, 42, args.device
    agent_cfg.device, agent_cfg.seed, agent_cfg.resume = args.device, 42, False

    fk, self_audit = configure_self_contacts(cfg, H1_2_CONTROL_JOINTS)
    report['front_guard_setup'] = self_audit

    class ComboEnv(H1_2StandV1Env):
        # Keep standing reset/action projection/rewards/termination. Restore actual
        # TT ball/prediction observations and physics; standing gets a separate view.
        def hold_standing_targets(self):
            pass

        def reset_ball(self, env_ids):
            return H1_2TTEnv.reset_ball(self, env_ids)

        def compute_intermediate_values(self):
            return H1_2TTEnv.compute_intermediate_values(self)

        def _h1_2_relative_target(self, obs):
            return H1_2TTEnv._h1_2_relative_target(self, obs)

        def compute_current_observations(self):
            return H1_2TTEnv.compute_current_observations(self)

        def compute_current_observations_perception(self):
            return H1_2TTEnv.compute_current_observations_perception(self)

        def check_reset(self):
            reset, timeout = super().check_reset()
            if recording:
                actual_q = self.robot.data.joint_pos[0].detach().cpu().numpy()
                pose_ok, info = arm_guard.inspect(actual_q, float(self.episode_length_buf[0])*self.step_dt >= 5.-1e-6)
                context['actual_guard'] = info
                if not pose_ok or context['self_contact_peak_N'] > .1:
                    stop_reasons.append(dict(time_s=float(self.episode_length_buf[0])*self.step_dt,
                                             actual_pose=info, self_contact_peak_N=context['self_contact_peak_N']))
                    reset = reset | torch.ones_like(reset, dtype=torch.bool)
                capture(self, reset, timeout)
            return reset, timeout

    def values(x):
        assert torch.isfinite(x).all(), 'NONFINITE PROBE STATE'
        return x.detach().cpu().tolist()

    def capture(e, reset, timeout):
        d = e.robot.data
        position = d.root_link_pos_w-e.scene.env_origins
        error = d.joint_pos_target-d.joint_pos
        estimate = kp*error+kd*(d.joint_vel_target-d.joint_vel)+d.joint_effort_target
        feet = e.contact_sensor.data.net_forces_w[:, e.feet_cfg.body_ids, 2]
        assert feet.shape == (1, 2)
        batch = {name: values(value)[0] for name, value in {
            'time_s': e.episode_length_buf*e.step_dt, 'height_m': position[:, 2],
            'tilt_rad': torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1., 1.)),
            'xy_drift_m': torch.linalg.vector_norm(position[:, :2]-e.stand_goal_env()[:, :2], dim=1),
            'speed_mps': torch.linalg.vector_norm(d.root_lin_vel_w, dim=1),
            'angular_speed_rps': torch.linalg.vector_norm(d.root_ang_vel_b, dim=1),
            'foot_fz_N': feet, 'tracking_error_rad': error, 'estimated_load_ratio': estimate.abs()/limits,
            'joint_actual_rad': d.joint_pos, 'joint_target_rad': d.joint_pos_target,
            'joint_velocity_radps': d.joint_vel, 'root_position_env_m': position,
            'ball_position_env_m': e.ball.data.root_pos_w-e.scene.env_origins,
            'ball_velocity_world_mps': e.ball.data.root_lin_vel_w,
            'paddle_position_env_m': e.paddle_pos, 'paddle_normal_world': e.paddle_normal,
            'pace_proximity_score': e.ball_contact, 'pace_latched_touch': e.has_touch_paddle,
            'projection_applied': (e.stand_projection_delta.abs() > 1e-6).any(dim=1),
            'reset': reset, 'timeout': timeout,
        }.items()}
        if not stats.ended:
            stats.add(batch)
            batch.update(reference_height_m=float(cfg.reference_base_height),
                         physical_failure=stats.physical_failure or bool(stop_reasons), arm_blend=context['blend'],
                         stand_actions=values(context['stand_actions'])[0],
                         tt_actions=values(context['tt_actions'])[0],
                         mixed_actions=values(context['mixed'])[0],
                         actual_front_guard=context['actual_guard'],
                         arm_target_guard=arm_guard.last_info,
                         self_contact_peak_N=context['self_contact_peak_N'])
            samples.append(batch)

    env = ComboEnv(cfg, args.headless)
    check_runtime(env)
    assert not cfg.domain_rand.perception_delay.enable and not env.add_noise
    assert env.num_envs == 1
    arm_guard, self_views = bind_guard(env, fk, self_audit)
    right = right_arm_indices(env.action_joint_names)
    report['controller_note'] = 'Both arm targets now rate/sweep filtered;13 leg/torso channels unchanged. First5s move to verified front ready pose.'
    report['unfiltered_standing_controlled_joint_names'] = [name for name in env.action_joint_names if '_hip_' in name or '_knee_' in name or '_ankle_' in name or name == 'torso_joint']
    report['guarded_arm_joint_names'] = [arm_guard.fk.names[j] for j in arm_guard.arms]
    report['tt_controlled_joint_names'] = [env.action_joint_names[i] for i in right]
    report['standing_controlled_joint_names'] = [name for i, name in enumerate(env.action_joint_names) if i not in right]
    report['joint_names'] = list(env.robot.joint_names)
    view = env.robot.root_physx_view
    kp = view.get_dof_stiffnesses().clone().to(env.device)
    kd = view.get_dof_dampings().clone().to(env.device)
    limits = view.get_dof_max_forces().clone().to(env.device)
    assert (limits > 0).all()
    stats = EpisodeMetrics(env.robot.joint_names, float(cfg.reference_base_height), env.step_dt)
    report.update(reference_height_m=float(cfg.reference_base_height), step_dt=env.step_dt,
                  physics_dt=env.sim.get_physics_dt(), runtime_kp=values(kp)[0], runtime_kd=values(kd)[0])

    # Use the same PhysX filtered contact API as IsaacLab2.1 ContactSensor. A
    # dedicated paddle->ball view avoids counting ground/self/other-body contacts.
    paddle_path = env.scene.env_prim_paths[0]+'/Robot/'+H1_2_PADDLE_BODY_NAME
    ball_path = str(env.ball.cfg.prim_path).replace('{ENV_REGEX_NS}', env.scene.env_prim_paths[0]).replace('env_.*', 'env_0')
    from pxr import UsdPhysics, PhysxSchema
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    assert stage.GetPrimAtPath(ball_path).HasAPI(UsdPhysics.RigidBodyAPI), 'BALL RIGID BODY PATH MISMATCH'
    assert stage.GetPrimAtPath(paddle_path).HasAPI(PhysxSchema.PhysxContactReportAPI), 'PADDLE CONTACT REPORTER MISSING'
    pair = env.contact_sensor._physics_sim_view.create_rigid_contact_view(paddle_path, filter_patterns=[ball_path])
    assert pair.filter_count == 1 and pair.get_contact_force_matrix(dt=env.sim.get_physics_dt()).numel() == 3
    report['physical_contact_paths'] = [paddle_path, ball_path]

    train_cfg = torch.load(str(RUN/'runner_config.pt'), map_location='cpu', weights_only=False)
    standing_runner = OnPolicyRunner(env, deepcopy(train_cfg), log_dir=None, device=args.device)
    standing_runner.load(str(checkpoint), load_optimizer=False)
    stand_policy = standing_runner.get_inference_policy(device=env.device)
    tt_runner = OnPolicyPredictorRegressionRunner(env, agent_cfg.to_dict(), log_dir=str(output/'tt_runner'), device=args.device)
    tt_runner.load(str(TT_CHECKPOINT), load_optimizer=False)
    tt_policy = tt_runner.get_inference_policy(device=env.device)
    modules = {'standing_policy': standing_runner.alg.policy,
               'standing_actor_normalizer': standing_runner.obs_normalizer,
               'standing_critic_normalizer': standing_runner.privileged_obs_normalizer,
               'tt_policy': tt_runner.alg.policy, 'tt_predictor': tt_runner._predictor}
    if isinstance(getattr(tt_runner, 'obs_normalizer', None), torch.nn.Module):
        modules['tt_actor_normalizer'] = tt_runner.obs_normalizer
    for module in modules.values():
        module.eval()
    states = {name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()} for name, module in modules.items()}
    saved_stand = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    saved_tt = torch.load(str(TT_CHECKPOINT), map_location='cpu', weights_only=False)
    keys = {'standing_policy': 'model_state_dict', 'standing_actor_normalizer': 'obs_norm_state_dict',
            'standing_critic_normalizer': 'privileged_obs_norm_state_dict', 'tt_policy': 'model_state_dict',
            'tt_predictor': 'pred_state_dict', 'tt_actor_normalizer': 'obs_norm_state_dict'}
    for name, state in states.items():
        if state:
            saved = saved_stand if name.startswith('standing_') else saved_tt
            assert state.keys() == saved[keys[name]].keys()
            assert all(torch.equal(value, saved[keys[name]][k].cpu()) for k, value in state.items()), 'LOADED STATE MISMATCH: '+name
    del saved_stand, saved_tt

    original_sim_step = env.sim.step
    def observed_physics_step(*positional, **keywords):
        result = original_sim_step(*positional, **keywords)
        if recording:
            context['substep'] += 1
            for source_name, self_view in zip(arm_guard.watch, self_views):
                self_force = self_view.get_contact_force_matrix(dt=env.sim.get_physics_dt())
                assert torch.isfinite(self_force).all()
                magnitude_self = float(torch.linalg.vector_norm(self_force.reshape(-1, 3), dim=1).max().item())
                context['self_contact_peak_N'] = max(context['self_contact_peak_N'], magnitude_self)
                if magnitude_self > .1:
                    self_contacts.append(dict(time_s=context['step']*env.step_dt+context['substep']*env.sim.get_physics_dt(),
                                              source_body=source_name, peak_pair_force_N=magnitude_self))
            force = pair.get_contact_force_matrix(dt=env.sim.get_physics_dt())
            assert torch.isfinite(force).all()
            magnitude = float(torch.linalg.vector_norm(force).item())
            if magnitude > .01:
                contacts.append({'time_s': context['step']*env.step_dt+context['substep']*env.sim.get_physics_dt(),
                                 'force_N': magnitude, 'force_world_N': values(force.reshape(3)),
                                 'arm_blend': context['blend']})
        return result
    env.sim.step = observed_physics_step
    if not args.headless:
        env.sim.set_camera_view((-4.5, -3.2, 2.15), (-1., 0., .95))
    scales = tuple(float(getattr(env.obs_scales, k)) for k in ('ball_pos', 'robot_pos', 'projected_gravity'))
    with torch.inference_mode():
        env.reset(torch.arange(1, device=env.device))
        env.compute_intermediate_values()
        obs, _ = env.get_observations()
        assert obs.shape == (1, 495) and (env.episode_length_buf == 0).all()
        ready_targets = obs.new_tensor(arm_guard.ready[[int(j) for j in env.action_joint_ids]]).reshape(1, -1)
        ready_actions = (ready_targets-env.robot.data.default_joint_pos[:, env.action_joint_ids])/env.action_scale
        assert (ready_actions[:, right].abs() <= env.clip_actions).all(), 'READY TARGET WOULD BE ACTION CLIPPED'
        recording = True
        wall_start = time.perf_counter()
        print('[HIT-BALANCE] NEW front-guarded mixed controller; real serves, self contacts enabled; no training/PD/asset edits.', flush=True)
        for step in range(int(env.max_episode_length)):
            if not args.headless and not app.is_running():
                raise RuntimeError('GUI CLOSED BEFORE TEST COMPLETED')
            stand_obs = standing_view(obs, env.stand_paddle_goal_env(), env._h1_2_reference_paddle_offset_b,
                                      cfg.ready_base_xy, scales, torch)
            # Independent current-pose check of the algebra, every control step.
            actual_xy = env.robot.data.root_link_pos_w[:, :2]-env.scene.env_origins[:, :2]
            expected_xy = (env.stand_goal_env()[:, :2]-actual_xy)*scales[1]
            assert torch.allclose(stand_obs.reshape(1, 5, 99)[:, -1, 96:98], expected_xy, atol=2e-4), 'STANDING OBS VIEW MISMATCH'
            assert torch.isfinite(stand_obs).all() and torch.isfinite(obs).all()
            stand_actions, tt_actions = stand_policy(stand_obs), tt_policy(obs)
            blend = arm_blend(step*env.step_dt)
            mixed = stand_actions.clone()
            # Blend from the verified front-ready residual, not the old default
            # pose. Clip the TT proposal before interpolation, as env.step does.
            tt_right = torch.clip(tt_actions[:, right], -env.clip_actions, env.clip_actions)
            mixed[:, right] = ready_actions[:, right]*(1.-blend)+tt_right*blend
            mixed = apply_guard(env, mixed, arm_guard, step*env.step_dt)
            assert mixed.shape == (1, 27) and torch.isfinite(mixed).all()
            context.update(step=step, substep=0, self_contact_peak_N=0., blend=blend, stand_actions=stand_actions.clone(),
                           tt_actions=tt_actions.clone(), mixed=mixed.clone())
            obs, reward, done, extras = env.step(mixed)
            assert abs(context['substep']*env.sim.get_physics_dt()-env.step_dt) < 1e-9
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            if not stats.ended:
                tt_runner._record_ball_positions()
                prediction_before = env.ball_prediction.detach().clone()
                tt_runner._maybe_predict_and_update_env()
                assert torch.isfinite(env.ball_prediction).all()
                report['predictor_calls'] += 1
                report['predictor_changed_calls'] += int(not torch.allclose(
                    prediction_before, env.ball_prediction, atol=1e-6, rtol=0.))
                samples[-1]['prediction_after_predictor_env_m'] = values(env.ball_prediction)[0]
            if not args.headless:
                env.sim.render()
                remaining = wall_start+(step+1)*env.step_dt-time.perf_counter()
                if remaining > 0:
                    time.sleep(min(remaining, env.step_dt))
            if stats.ended:
                break
        recording = False
    for name, module in modules.items():
        assert all(torch.equal(v.cpu(), states[name][k]) for k, v in module.state_dict().items()), 'MODEL STATE CHANGED: '+name
    assert sha(checkpoint) == training['checkpoint_sha256'] and sha(TT_CHECKPOINT) == tt_sha
    verify_originals(originals)
    verify_probe()
    verify_urdf(self_audit)
    events = contact_windows(samples, contacts, env.step_dt)
    standing = stats.result()
    standing['self_collision_or_workspace_failure'] = bool(stop_reasons)
    standing['stage1_candidate_pass'] = standing['stage1_candidate_pass'] and not stop_reasons
    verdict = ('NO_ACTIVE_PHYSICAL_CONTACT_OBSERVED' if not events else
               'BALANCE_SCREEN_FAILED' if not standing['stage1_candidate_pass'] or any(
                   e['post_window_complete'] and not e['post_window_body_support_pass'] for e in events) else
               'POST_CONTACT_WINDOW_INCOMPLETE' if any(not e['post_window_complete'] for e in events) else
               'CONTACT_WITH_BALANCE_SCREEN_PASS' if all(e['post_window_body_support_pass'] for e in events)
               and standing['stage1_candidate_pass'] else 'BALANCE_SCREEN_FAILED')
    report.update(front_guard_calls=arm_guard.calls, restricted_calls=arm_guard.restricted,
                  active_guard_calls=arm_guard.active_calls, active_restricted_calls=arm_guard.active_restricted,
                  self_contacts=self_contacts, front_guard_stop_reasons=stop_reasons)
    report.update(completed=True, standing=standing, active_physical_contact_events=events, verdict=verdict,
                  physical_contact_substep_observations=len(contacts), inference_model_states_unchanged=True,
                  original_files_unchanged=True)
    print('[HIT-BALANCE] RESULT:', verdict, 'duration=', standing['duration_s'],
          'standing_pass=', standing['stage1_candidate_pass'], 'active_contact_events=', len(events), flush=True)
    print('[HIT-BALANCE] BODY: min_z=', standing['min_height_m'], 'max_tilt=', standing['max_tilt_rad'],
          'max_drift=', standing['max_xy_drift_m'], 'both_feet_fraction=', standing['both_feet_contact_fraction'],
          'estimated_load_flags=', standing['estimated_load_flag_joints'], flush=True)
    print('[FRONT-GUARD] RESULT: body_pass=', standing['stage1_candidate_pass'], 'self_contact_samples=', len(self_contacts),
          'stop_reasons=', stop_reasons, 'restricted_calls=', arm_guard.restricted, '/', arm_guard.calls,
          'active_restricted_calls=', arm_guard.active_restricted, '/', arm_guard.active_calls, flush=True)
    active_rows = [r for r in samples if r['time_s'] >= 5.-1e-6]
    if active_rows:
        print('[FRONT-GUARD] ACTIVE_CLEARANCE: paddle_front_min_m=', min(r['actual_front_guard']['paddle_front_clearance_m'] for r in active_rows),
              'wrist_front_min_m=', min(r['actual_front_guard']['wrist_front_clearance_m'] for r in active_rows),
              'box_axis_gap_min_m=', min(r['actual_front_guard']['min_box_separation_axis_gap_m'] for r in active_rows), flush=True)
    print('[HIT-BALANCE] CONTACT_WINDOWS:', events, flush=True)
    print('[HIT-BALANCE] PREDICTOR: calls=', report['predictor_calls'],
          'changed_calls=', report['predictor_changed_calls'],
          '; change is not an accuracy test; internal legacy update may suppress exceptions.', flush=True)
except BaseException as error:
    recording = False
    report.update(completed=False, error=repr(error), traceback=traceback.format_exc(),
                  self_contacts=self_contacts, front_guard_stop_reasons=stop_reasons)
    if stats is not None and stats.count:
        report['partial_standing'] = stats.result()
    traceback.print_exc()
    raise
finally:
    recording = False
    try:
        if env is not None and original_sim_step is not None:
            env.sim.step = original_sim_step
        write_json(output/'hitting_front_report.json', report)
        write_json(output/'full_trace.json', {'control_steps': samples, 'physical_contact_substeps': contacts, 'self_contact_substeps': self_contacts})
        print('[HIT-BALANCE] REPORT:', output/'hitting_front_report.json', flush=True)
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if app is not None:
                app.close()
