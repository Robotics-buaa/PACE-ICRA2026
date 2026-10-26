"""One first-episode H1-2 stance trace; no model, learning, or parameter edits.

Torque values are estimates, not measured PhysX joint drive forces.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path('/workspace/projects/PACE-ICRA2026')
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=float, default=2.5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0 < args.seconds <= 3:
    parser.error('--seconds must be in (0, 3]')
app = AppLauncher(args).app
env = None

try:
    import torch
    import isaaclab.utils.math as math_utils
    import legged_lab.envs
    from legged_lab.utils.task_registry import task_registry

    manifest = json.loads((ROOT / 'tools/h1_2/bilateral_install_report.json').read_text())
    protected = {**manifest['protected_sha256'], **manifest['new_files_sha256']}

    def verify():
        for name, expected in protected.items():
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            assert actual == expected, 'SOURCE CHANGED: ' + name

    verify()
    cfg, _ = task_registry.get_cfgs('h1_2_tt_bilateral')
    cfg = deepcopy(cfg)
    cfg.scene.num_envs = 1
    cfg.scene.seed = 42
    cfg.device = args.device
    cfg.noise.add_noise = False
    cfg.domain_rand.action_delay.enable = False
    cfg.domain_rand.perception_delay.enable = False
    cfg.domain_rand.events.push_robot = None
    cfg.domain_rand.events.add_base_mass = None
    cfg.domain_rand.events.physics_material = None
    cfg.domain_rand.events.reset_base.params['pose_range'] = {
        key: (0.0, 0.0) for key in ('x', 'y', 'yaw')
    }
    cfg.domain_rand.events.reset_base.params['velocity_range'] = {
        key: (0.0, 0.0) for key in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')
    }
    cfg.domain_rand.events.reset_locomotion_joints.params['position_range'] = (1.0, 1.0)
    cfg.domain_rand.events.reset_manipulation_joints.params['position_range'] = (0.0, 0.0)
    for event in (cfg.domain_rand.events.reset_locomotion_joints,
                  cfg.domain_rand.events.reset_manipulation_joints):
        if 'velocity_range' in event.params:
            event.params['velocity_range'] = (0.0, 0.0)

    rows = []
    recording = False
    runtime = None
    Parent = task_registry.get_task_class('h1_2_tt_bilateral')

    class TrackingFixture(Parent):
        def reset_ball(self, env_ids):
            super().reset_ball(env_ids)
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            if len(ids):
                state = self.ball.data.default_root_state[ids].clone()
                state[:, :3] = self.scene.env_origins[ids] + state.new_tensor([100., 100., 1000.])
                state[:, 7:] = 0.0
                self.reset_ball_state_buf[ids] = state
                self.ball.write_root_pose_to_sim(state[:, :7], ids)
                self.ball.write_root_velocity_to_sim(state[:, 7:], ids)

        def reset(self, env_ids):
            super().reset(env_ids)
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            if not len(ids):
                return
            state = self.robot.data.default_root_state[ids].clone()
            state[:, :3] += self.scene.env_origins[ids]
            state[:, 0] = self.scene.env_origins[ids, 0] - 1.6
            state[:, 1] = self.scene.env_origins[ids, 1]
            state[:, 7:] = 0.0
            self.robot.write_root_pose_to_sim(state[:, :7], ids)
            self.robot.write_root_velocity_to_sim(state[:, 7:], ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.compute_perception()
            self.compute_paddle_touch()

        def check_reset(self):
            reset, timeout = super().check_reset()
            if recording:
                rows.append(snapshot(self, reset, timeout))
            return reset, timeout

    def vec(value):
        assert bool(torch.isfinite(value).all()), 'NONFINITE TRACE VALUE'
        return value.detach().cpu().tolist()

    def snapshot(e, reset=None, timeout=None):
        d = e.robot.data
        q, v = d.joint_pos[0], d.joint_vel[0]
        qt, vt, ff = d.joint_pos_target[0], d.joint_vel_target[0], d.joint_effort_target[0]
        # Same-state P/D decomposition; no claim of measured solver torque.
        p = runtime['kp_tensor'] * (qt - q)
        damp = runtime['kd_tensor'] * (vt - v)
        pd = p + damp + ff
        mass = runtime['mass_tensor']
        com = (d.body_com_pos_w[0] * mass[:, None]).sum(0) / mass.sum()
        feet = d.body_link_pos_w[0, runtime['foot_ids']]
        raw = math_utils.euler_xyz_from_quat(d.root_link_quat_w)[1][0]
        pitch = torch.remainder(raw + torch.pi, 2 * torch.pi) - torch.pi
        result = {
            'time_s': float(e.episode_length_buf[0].item()) * e.step_dt,
            'base_z': float((d.root_link_pos_w[0, 2] - e.scene.env_origins[0, 2]).item()),
            'pitch_rad': float(pitch.item()),
            'tilt_rad': float(torch.acos((-d.projected_gravity_b[0, 2]).clamp(-1, 1)).item()),
            'q': vec(q), 'q_target': vec(qt), 'q_minus_target': vec(q - qt),
            'qd': vec(v), 'qd_target': vec(vt), 'effort_feedforward': vec(ff),
            'p_estimate_Nm': vec(p), 'd_estimate_Nm': vec(damp),
            'pd_plus_ff_estimate_Nm': vec(pd),
            'pd_estimate_abs_over_runtime_effort_limit': vec(pd.abs() / runtime['limit_tensor']),
            'actuator_computed_estimate_cached_Nm': vec(d.computed_torque[0]),
            'actuator_clipped_estimate_cached_Nm': vec(d.applied_torque[0]),
            'whole_body_com_w': vec(com),
            'whole_body_com_minus_foot_link_mean_w': vec(com - feet.mean(0)),
            'pelvis_minus_foot_link_mean_x': float((d.root_link_pos_w[0, 0] - feet[:, 0].mean()).item()),
            'foot_link_positions_w': vec(feet),
            'feet_forces_w': vec(e.contact_sensor.data.net_forces_w[0, e.feet_cfg.body_ids]),
            'reset': bool(reset[0].item()) if reset is not None else False,
            'timeout': bool(timeout[0].item()) if timeout is not None else False,
        }
        return result

    with torch.no_grad():
        env = TrackingFixture(cfg, headless=args.headless)
        assert not env.robot.is_fixed_base
        assert not cfg.scene.robot.spawn.rigid_props.disable_gravity
        assert all(a.is_implicit_model for a in env.robot.actuators.values()), 'EXPECTED IMPLICIT ACTUATORS'
        view = env.robot.root_physx_view
        kp = view.get_dof_stiffnesses()[0].clone().to(env.device)
        kd = view.get_dof_dampings()[0].clone().to(env.device)
        limits = view.get_dof_max_forces()[0].clone().to(env.device)
        masses = view.get_masses()[0].clone().to(env.device)
        assert bool((limits > 0).all()) and bool((masses >= 0).all()) and masses.sum() > 0
        assert len(masses) == env.robot.data.body_com_pos_w.shape[1]
        foot_ids, foot_names = env.robot.find_bodies(
            ['left_ankle_roll_link', 'right_ankle_roll_link'], preserve_order=True)
        assert len(foot_ids) == 2
        runtime = {'kp_tensor': kp, 'kd_tensor': kd, 'limit_tensor': limits,
                   'mass_tensor': masses, 'foot_ids': foot_ids}
        joint_names = list(env.robot.joint_names)
        assert len(joint_names) == 27
        lower_names = [f'{side}_{part}_joint' for side in ('left', 'right')
                       for part in ('hip_pitch', 'knee', 'ankle_pitch')]
        lower_ids = [joint_names.index(n) for n in lower_names]
        print('[TRACK] RUNTIME: joint Kp Kd effort_limit_Nm', flush=True)
        for n, j in zip(lower_names, lower_ids):
            print(n, float(kp[j]), float(kd[j]), float(limits[j]), flush=True)
        print('[TRACK] articulation mass_kg:', float(masses.sum()), flush=True)
        initial = snapshot(env)
        recording = True
        for _ in range(max(1, round(args.seconds / env.step_dt))):
            obs, reward, done, extras = env.step(torch.zeros((1, 27), device=env.device))
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            if rows and rows[-1]['reset']:
                break
        recording = False
    verify()
    assert rows, 'NO PRE-RESET TRACE SAMPLES'
    report = {
        'case': 'x=-1.60, ball far away; single environment, first episode only',
        'requested_seconds': args.seconds, 'step_dt': env.step_dt,
        'joint_names': joint_names, 'body_names': list(env.robot.body_names),
        'runtime_kp': vec(kp), 'runtime_kd': vec(kd), 'runtime_effort_limit_Nm': vec(limits),
        'runtime_body_masses_kg': vec(masses), 'foot_link_names': foot_names,
        'contact_sensor_feet_body_ids': vec(torch.as_tensor(env.feet_cfg.body_ids)),
        'source_sha256_verified': protected,
        'neutral_actions': True, 'model_loaded': False, 'learning_iterations': 0,
        'gains_pose_limits_assets_modified': False,
        'sampling': 'control-rate check_reset, before automatic reset; initial separately before first action',
        'torque_semantics': 'PD estimates only; cached actuator estimates may precede the sampled state. No measured PhysX joint drive force.',
        'com_semantics': 'mass-weighted COM of all articulation bodies; foot link mean is only a reference, not COP or a support polygon',
        'initial_before_first_action': initial, 'samples': rows,
    }
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = ROOT / 'tools/h1_2' / ('stance_tracking_report_' + stamp + '.json')
    with out.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('[TRACK] t z pitch COM-minus-foot-X knee_error(L,R) knee_qd(L,R) knee_PD_est/limit(L,R)', flush=True)
    knees = [joint_names.index(side + '_knee_joint') for side in ('left', 'right')]
    chosen = {0, len(rows) - 1}
    for t in (0.2, 0.5, 0.8, 1.0, 1.2):
        if t <= rows[-1]['time_s']:
            chosen.add(min(range(len(rows)), key=lambda i: abs(rows[i]['time_s'] - t)))
    first_tilt = next((i for i, r in enumerate(rows) if r['tilt_rad'] > 0.25), None)
    if first_tilt is not None:
        chosen.add(first_tilt)
    for i in sorted(chosen):
        r = rows[i]
        pairs = [[round(r[k][j], 4) for j in knees]
                 for k in ('q_minus_target', 'qd', 'pd_estimate_abs_over_runtime_effort_limit')]
        print('[TRACK]', round(r['time_s'], 3), round(r['base_z'], 4), round(r['pitch_rad'], 4),
              round(r['whole_body_com_minus_foot_link_mean_w'][0], 4), *pairs, flush=True)
    selected = rows[first_tilt] if first_tilt is not None else rows[-1]
    print('[TRACK] DETAIL AT', selected['time_s'], '(first tilt>0.25, else last sample)', flush=True)
    print('[TRACK] joint q target qd P_est D_est FF PD_est limit', flush=True)
    for n, j in zip(lower_names, lower_ids):
        values = [selected[key][j] for key in ('q', 'q_target', 'qd', 'p_estimate_Nm',
                  'd_estimate_Nm', 'effort_feedforward', 'pd_plus_ff_estimate_Nm')]
        print(n, *[round(x, 4) for x in values], round(float(limits[j]), 4), flush=True)
    print('[TRACK] feet_forces_w:', selected['feet_forces_w'], flush=True)
    print('[TRACK] max abs PD_est/limit before tilt (or all if no tilt):', flush=True)
    early = rows[:first_tilt + 1] if first_tilt is not None else rows
    for n, j in zip(lower_names, lower_ids):
        peak = max(r['pd_estimate_abs_over_runtime_effort_limit'][j] for r in early)
        print(n, round(peak, 4), flush=True)
    print('Report:', out, flush=True)
    print('H1-2 STANCE TRACKING CAPTURE FINISHED (not a standing pass)', flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    try:
        if env is not None:
            env.close()
    finally:
        app.close()
