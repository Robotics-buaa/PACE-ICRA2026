"""Independent 7-action task around an unchanged 27-action standing interface.

Import after AppLauncher. Physical ready reset is independent of default_joint_pos.
One serve per episode; terminal evidence/reward is captured before any reset.
"""
from copy import deepcopy
import numpy as np
import torch

from paths import RUN, STAND_SHA, sha
from batch_geometry import BatchGeometry, quaternion_matrix
from target_guard import choose_targets
from standing_v1.task import H1_2StandV1Env, make_cfg as stand_cfg
from legged_lab.envs.h1_2_tt.h1_2_tt_env import H1_2TTEnv
from legged_lab.assets.unitree.h1_2_tt import H1_2_CONTROL_JOINTS
from runtime import configure_self_contacts, bind_guard
from ready_pose import RIGHT_READY_RAD

ARM_NAMES = tuple(RIGHT_READY_RAD)
HARD = (.01, .03, .02)
# Initial conservative curriculum, retained from the validated probe. These are
# not demonstrated final hitting speeds. Performance must be measured before expansion.
LOOKAHEAD_S = .08
FRAME = 91
HISTORY = 5


def make_cfg(num_envs, device, seed=42, acceptance=False):
    from legged_lab.utils import task_registry
    cfg = stand_cfg()
    tt, _ = task_registry.get_cfgs('h1_2_tt')
    cfg.ball = deepcopy(tt.ball)
    cfg.ball.ball_reset_repeat = 1
    cfg.scene.num_envs, cfg.scene.seed, cfg.device = num_envs, seed, device
    cfg.scene.max_episode_length_s = 20. if acceptance else 4.
    # The ball is the sensor for actual paddle/table impulses.
    cfg.scene.ball.spawn.activate_contact_sensors = True
    fk, audit = configure_self_contacts(cfg, H1_2_CONTROL_JOINTS)
    return cfg, fk, audit


class ReadyPlant(H1_2StandV1Env):
    """Keep old defaults, reference geometry, observation order/scales and gains."""
    def init_buffers(self):
        super().init_buffers()
        self.ready_q = self.robot.data.default_joint_pos.clone()
        for name, value in RIGHT_READY_RAD.items():
            self.ready_q[:, self.robot.joint_names.index(name)] = value
        self.applied27 = torch.zeros((self.num_envs, 27), device=self.device)
        self._ready_residual = ((self.ready_q[:, self.action_joint_ids]
                                - self.robot.data.default_joint_pos[:, self.action_joint_ids])/self.action_scale)
        assert (self._ready_residual.abs() <= self.clip_actions).all(), 'READY RESIDUAL WOULD BE CLIPPED'

    def reset_ball(self, env_ids):
        return H1_2TTEnv.reset_ball(self, env_ids)

    def compute_current_observations(self):
        actor, critic = super().compute_current_observations()
        if hasattr(self, 'applied27'):
            actor[:, 60:87] = self.applied27*self.obs_scales.actions
            critic[:, 60:87] = self.applied27*self.obs_scales.actions
        return actor, critic

    def compute_current_observations_perception(self):
        # Noise and perception delay are explicitly disabled by the task config.
        return self.compute_current_observations()

    def reset(self, env_ids):
        super().reset(env_ids)
        if not len(env_ids) or not hasattr(self, 'ready_q'):
            return
        q = self.ready_q[env_ids].clone()
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        self.robot.set_joint_velocity_target(torch.zeros_like(q), env_ids=env_ids)
        self.robot.set_joint_effort_target(torch.zeros_like(q), env_ids=env_ids)
        self.applied27[env_ids] = self._ready_residual[env_ids]
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.compute_perception()
        self.compute_paddle_touch()
        self.hold_standing_targets()


class ContactPairs:
    """One sensor per explicit env path, filter paths paired to that same env."""
    def __init__(self, env, watch):
        self.env = env
        sim = env.contact_sensor._physics_sim_view
        roots = [p+'/Robot/' for p in env.scene.env_prim_paths]
        sensors = [r+n for r in roots for n in watch]
        filters = [[r+'torso_link', r+'pelvis'] for r in roots for n in watch]
        self.body = sim.create_rigid_contact_view(sensors, filter_patterns=filters)
        self.order_body = self._order(self.body, sensors, 2)
        balls = [p+'/Ball' for p in env.scene.env_prim_paths]
        from pxr import Usd, UsdPhysics
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        table_paths = []
        for p in env.scene.env_prim_paths:
            bodies = [str(b.GetPath()) for b in Usd.PrimRange(stage.GetPrimAtPath(p+'/Table'))
                      if b.HasAPI(UsdPhysics.RigidBodyAPI)]
            assert len(bodies) == 1, 'EXPECTED ONE TABLE RIGID BODY PER ENV'
            table_paths.append(bodies[0])
        targets = [[r+'h1_2_paddle_link', table] for r, table in zip(roots, table_paths)]
        self.ball = sim.create_rigid_contact_view(balls, filter_patterns=targets)
        self.order_ball = self._order(self.ball, balls, 2)
        self.nwatch = len(watch)

    def _order(self, view, expected, nfilter):
        assert view.sensor_count == len(expected) and view.filter_count == nfilter, 'CONTACT VIEW SHAPE MISMATCH'
        # Binding order can be lexical (env_10 before env_2). Never assume it.
        paths = list(view.sensor_paths)
        assert len(set(paths)) == len(paths) and set(paths) == set(expected), 'CONTACT SENSOR PATH MISMATCH'
        return torch.tensor([paths.index(p) for p in expected], device=self.env.device)

    def read(self):
        dt = self.env.physics_dt
        b = self.body.get_contact_force_matrix(dt=dt)[self.order_body]
        f = self.ball.get_contact_force_matrix(dt=dt)[self.order_ball]
        assert b.shape == (self.env.num_envs*self.nwatch, 2, 3)
        assert f.shape == (self.env.num_envs, 2, 3)
        self.table_force_z = f[:, 1, 2].abs()
        return torch.linalg.vector_norm(b, dim=-1).reshape(self.env.num_envs, -1).amax(-1), torch.linalg.vector_norm(f, dim=-1)


class ReadyTask:
    """rsl_rl VecEnv-like wrapper. Its policy owns seven channels, never 27."""
    def __init__(self, cfg, fk, audit, headless, acceptance=False):
        self.e = ReadyPlant(cfg, headless)
        self.cfg, self.device = cfg, self.e.device
        self.num_envs, self.num_actions = self.e.num_envs, 7
        self.num_obs = self.num_privileged_obs = FRAME*HISTORY
        self.step_dt = self.e.step_dt
        self.max_episode_length = int(self.e.max_episode_length)
        self.episode_length_buf = self.e.episode_length_buf
        self.acceptance = acceptance
        e = self.e
        self.audit = audit
        guard, unused_single_env_views = bind_guard(e, fk, audit)
        from scene_audit import verify_clones
        verify_clones(e, guard, audit)
        self._validation_views = unused_single_env_views
        self.geom = BatchGeometry(guard.fk, guard.bounds, guard.watch, torch,
                                  lambda a: torch.as_tensor(a, device=self.device, dtype=torch.float32))
        self.watch = guard.watch
        self.body_ids = [e.robot.body_names.index(n) for n in self.geom.names]
        self.arms = guard.arms
        self.right = [e.robot.joint_names.index(n) for n in ARM_NAMES]
        self.right_action = [e.action_joint_names.index(n) for n in ARM_NAMES]
        self.checked_joints = self.arms + [e.robot.joint_names.index('torso_joint')]
        self.hard = torch.tensor(HARD, device=self.device)
        self.inner = self.hard + torch.tensor([.005, .02, .015], device=self.device)
        self.safe_distance = torch.tensor([.04, .09, .06], device=self.device)
        self.rate = torch.tensor([.9 if 'wrist' in e.robot.joint_names[j] else .6 for j in self.arms], device=self.device)
        self.action_range = torch.tensor([.45, .35, .45, .45, .60, .50, .50], device=self.device)
        self.contacts = ContactPairs(e, self.watch)
        self.last_target = e.ready_q.clone()
        self.last_action = torch.zeros((self.num_envs, 7), device=self.device)
        self.filtered = torch.zeros(self.num_envs, device=self.device)
        self.no_path = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.flags = {k: torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) for k in
                      ('own_bounce', 'hit', 'valid_hit', 'landed', 'success', 'table_touch', 'paddle_touch')}
        self.stand_history = torch.zeros((self.num_envs, 5, 99), device=self.device)
        self.stand_critic = torch.zeros((self.num_envs, 5, 116), device=self.device)
        self.history = torch.zeros((self.num_envs, HISTORY, FRAME), device=self.device)
        self.extras = {}
        self.total = {k: 0 for k in ('episodes', 'failures', 'contacts', 'returns', 'guard_stops')}
        self._seed_histories(torch.arange(self.num_envs, device=self.device))
        # Bootstrap runner with the unchanged 27-action/495/580 standing interface.
        from rsl_rl.runners import OnPolicyRunner
        assert sha(RUN/'model_499.pt') == STAND_SHA
        train_cfg = torch.load(str(RUN/'runner_config.pt'), map_location='cpu', weights_only=False)
        self.frozen = OnPolicyRunner(e, deepcopy(train_cfg), log_dir=None, device=self.device)
        self.frozen.load(str(RUN/'model_499.pt'), load_optimizer=False)
        self.frozen_modules = [self.frozen.alg.policy, self.frozen.obs_normalizer, self.frozen.privileged_obs_normalizer]
        saved = torch.load(str(RUN/'model_499.pt'), map_location='cpu', weights_only=False)
        keys = ('model_state_dict', 'obs_norm_state_dict', 'privileged_obs_norm_state_dict')
        for module, key in zip(self.frozen_modules, keys):
            module.eval()
            module.requires_grad_(False)
            actual = module.state_dict()
            if actual:
                assert actual.keys() == saved[key].keys()
                assert all(torch.equal(v.cpu(), saved[key][k].cpu()) for k, v in actual.items()), 'FROZEN LOAD MISMATCH'
        self.frozen_states = [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in self.frozen_modules]
        self.stand_policy = self.frozen.get_inference_policy(device=self.device)
        self.feedback_calls = 0
        self.initial_defaults = e.robot.data.default_joint_pos.clone()
        self.initial_ref = e._h1_2_reference_paddle_offset_b.clone()
        self.initial_kp = e.robot.root_physx_view.get_dof_stiffnesses().clone()
        self.initial_kd = e.robot.root_physx_view.get_dof_dampings().clone()
        self._check_initial_geometry()

    def _check_initial_geometry(self):
        e = self.e
        actual = self.actual_margins()
        predicted = self.geom.inspect(e.robot.data.joint_pos)
        assert torch.allclose(actual, predicted, atol=.002, rtol=0.), 'BATCH FK/LIVE GEOMETRY MISMATCH'
        assert (actual >= self.hard).all(), 'READY INITIAL COLLISION OR WORKSPACE FAILURE'
        assert (e.robot.data.joint_vel.abs() < 1e-6).all(), 'READY INITIAL VELOCITY NOT ZERO'
        assert torch.allclose(e.robot.data.joint_pos_target, e.ready_q, atol=1e-6)
        assert not e.robot.is_fixed_base and not self.cfg.scene.robot.spawn.rigid_props.disable_gravity
        assert not e.add_noise and not self.cfg.domain_rand.action_delay.enable

    def actual_margins(self):
        d = self.e.robot.data
        return self.geom.measure(d.body_link_pos_w[:, self.body_ids],
                                 quaternion_matrix(d.body_link_quat_w[:, self.body_ids], torch))

    def body_values(self):
        e, d = self.e, self.e.robot.data
        root = d.root_link_pos_w-e.scene.env_origins
        h = root[:, 2]
        tilt = torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1., 1.))
        drift = torch.linalg.vector_norm(root[:, :2]-e.stand_goal_env()[:, :2], dim=1)
        foot = e.contact_sensor.data.net_forces_w[:, e.feet_cfg.body_ids, 2]
        load = (d.joint_stiffness*(d.joint_pos_target-d.joint_pos) + d.joint_damping*(d.joint_vel_target-d.joint_vel) + d.joint_effort_target).abs()/d.joint_effort_limits.clamp_min(1e-6)
        return h, tilt, drift, foot, load.amax(-1)

    def frame(self):
        e, d = self.e, self.e.robot.data
        opp_x = self.cfg.table.table_opponent_contact_x
        opp_y = self.cfg.table.table_opponent_contact_y
        opp_z = self.cfg.table.table_opponent_contact_z
        goal = d.joint_pos.new_tensor([(opp_x[0]+opp_x[1])*.5, (opp_y[0]+opp_y[1])*.5, (opp_z[0]+opp_z[1])*.5])
        out = torch.cat(((d.joint_pos-e.ready_q)[:, e.obs_joint_ids],
                         d.joint_vel[:, e.obs_joint_ids]*.1, d.root_ang_vel_b*.25,
                         d.projected_gravity_b, d.root_lin_vel_b*.5,
                         d.root_link_pos_w-e.scene.env_origins-e.stand_goal_env(),
                         e.ball.data.root_pos_w-e.paddle_touch_point,
                         e.ball.data.root_lin_vel_w*.1, e.paddle_normal,
                         self.actual_margins(), torch.stack((self.flags['own_bounce'], self.flags['hit']), -1).float(),
                         (e.episode_length_buf*self.step_dt*.05).unsqueeze(-1),
                         self.last_action, goal[None, :]-e.paddle_pos), -1)
        assert out.shape == (self.num_envs, FRAME), 'NEW OBSERVATION LAYOUT MISMATCH'
        assert torch.isfinite(out).all(), 'NONFINITE NEW OBSERVATION'
        return out.clamp(-20., 20.)

    def _seed_histories(self, ids):
        actor, critic = self.e.compute_current_observations()
        self.stand_history[ids] = actor[ids, None, :]
        self.stand_critic[ids] = critic[ids, None, :]
        self.history[ids] = self.frame()[ids, None, :]

    def _advance_histories(self):
        actor, critic = self.e.compute_current_observations()
        self.stand_history = torch.cat((self.stand_history[:, 1:], actor[:, None]), 1)
        self.stand_critic = torch.cat((self.stand_critic[:, 1:], critic[:, None]), 1)
        self.history = torch.cat((self.history[:, 1:], self.frame()[:, None]), 1)

    def get_observations(self):
        obs = self.history.reshape(self.num_envs, -1)
        return obs, {'observations': {'critic': obs}}

    def reset(self, ids):
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        if not len(ids):
            return
        self.e.reset(ids)
        self.last_target[ids] = self.e.ready_q[ids]
        self.last_action[ids] = 0.
        self.filtered[ids] = 0.
        self.no_path[ids] = False
        for flag in self.flags.values():
            flag[ids] = False
        self._seed_histories(ids)

    def select_targets(self, actions):
        e, d = self.e, self.e.robot.data
        with torch.no_grad():
            standing = self.stand_policy(self.stand_history.reshape(self.num_envs, 495))
        self.feedback_calls += 1
        target = d.joint_pos.clone()
        target[:, e.action_joint_ids] = standing.clamp(-e.clip_actions, e.clip_actions)*e.action_scale + d.default_joint_pos[:, e.action_joint_ids]
        command = actions.clamp(-1., 1.)
        target[:, self.right] = e.ready_q[:, self.right]+command*self.action_range
        # Both arms retain the same validated conservative speed limits.
        delta = (target[:, self.arms]-self.last_target[:, self.arms]).clamp(-self.rate*self.step_dt, self.rate*self.step_dt)
        target[:, self.arms] = self.last_target[:, self.arms]+delta
        target = torch.maximum(torch.minimum(target, d.joint_pos_limits[:, :, 1]), d.joint_pos_limits[:, :, 0])
        # Tracking error sets sweep density rather than a fixed five-point span.
        # Rollback interpolates from previous PD targets, preserving zero-action
        # ready references and the validated target-rate caps.
        desired = target.clone()
        selected, found, self.guard_info = choose_targets(
            d.joint_pos, d.joint_vel, self.last_target, desired, self.arms,
            self.checked_joints, self.rate, self.step_dt, self.geom, self.inner,
            LOOKAHEAD_S, torch, lambda x: target.new_tensor(x), self.hard, e.ready_q)
        # PhysX tensors may share cached storage. Snapshot BEFORE stepping.
        self.guard_info = {key: value.detach().clone() for key, value in self.guard_info.items()}
        self.no_path = ~found
        # No-safe-path environments receive only an actual-pose hold target,
        # remain marked failed, and are reset after terminal evidence is captured.
        self.filtered = (selected[:, self.arms]-desired[:, self.arms]).square().mean(-1)
        return selected, command

    def region(self, side, pos):
        result = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        for axis, key in enumerate(('x', 'y', 'z')):
            low, high = getattr(self.cfg.table, 'table_'+side+'_contact_'+key)
            result &= (pos[:, axis] >= low) & (pos[:, axis] <= high)
        return result

    @torch.no_grad()
    def step(self, actions):
        e, d = self.e, self.e.robot.data
        assert actions.shape == (self.num_envs, 7) and torch.isfinite(actions).all()
        target, command = self.select_targets(actions)
        early = self.no_path.clone()
        # Rejected candidates are never sent. Hold actual pose while the other
        # environments finish the batch step, then capture and reset failed rows.
        target = torch.where(early[:, None], d.joint_pos.clone(), target)
        self.last_target = target.clone()
        e.applied27 = (target[:, e.action_joint_ids]-d.default_joint_pos[:, e.action_joint_ids])/e.action_scale
        action_cost = (command-self.last_action).square().mean(-1)
        self.last_action = command.clone()
        contact_reward = torch.zeros(self.num_envs, device=self.device)
        return_reward = torch.zeros_like(contact_reward)
        direction_reward = torch.zeros_like(contact_reward)
        body_peak = torch.zeros_like(contact_reward)
        ball_peak = torch.zeros_like(contact_reward)
        margins = self.actual_margins()
        min_margins = margins.clone()
        unsafe = early.clone()
        for substep in range(self.cfg.sim.decimation):
            previous_vel = e.ball.data.root_lin_vel_w.clone()
            e.robot.set_joint_position_target(target)
            e.aero.apply_to_rigid_object(e.ball)
            e.scene.write_data_to_sim()
            e.sim.step(render=False)
            e.scene.update(dt=e.physics_dt)
            e.sim_step_counter += 1
            e.compute_perception()
            e.compute_paddle_touch()
            body_force, ball_force = self.contacts.read()
            assert torch.isfinite(body_force).all() and torch.isfinite(ball_force).all()
            body_peak = torch.maximum(body_peak, body_force)
            ball_peak = torch.maximum(ball_peak, ball_force[:, 0])
            margins = self.actual_margins()
            min_margins = torch.minimum(min_margins, margins)
            unsafe |= (body_force > .1) | (margins < self.hard).any(-1) | ~torch.isfinite(margins).all(-1)
            pos = e.ball.data.root_pos_w-e.scene.env_origins
            v = e.ball.data.root_lin_vel_w
            paddle = ball_force[:, 0] > .01
            # Table-top bounce requires a measured predominantly vertical
            # impulse and downward arrival; a net/edge force alone earns no landing.
            table = (ball_force[:, 1] > .01) & (self.contacts.table_force_z >= .8*ball_force[:, 1]) & (previous_vel[:, 2] < -.01)
            new_paddle = paddle & ~self.flags['paddle_touch'] & ~self.flags['hit'] & ~unsafe
            double_hit = paddle & ~self.flags['paddle_touch'] & self.flags['hit'] & ~unsafe
            hit_before = self.flags['hit'].clone()
            own = self.region('own', pos)
            opposite = self.region('opponent', pos)
            new_table = table & ~self.flags['table_touch'] & ~unsafe
            self.flags['own_bounce'] |= new_table & own & ~hit_before
            valid = new_paddle & self.flags['own_bounce'] & (previous_vel[:, 0] < -.1)
            self.flags['hit'] |= new_paddle
            self.flags['valid_hit'] |= valid
            self.flags['valid_hit'] &= ~double_hit
            contact_reward += 4.*valid.float()-4.*(new_paddle & ~valid).float()
            contact_reward -= 4.*double_hit.float()
            outgoing = v[:, 0] > .2
            direction_reward += valid.float()*outgoing.float()*2.*(v[:, 0]/6.).clamp(0., 1.)
            first_landing = new_table & hit_before & ~self.flags['landed']
            success = first_landing & opposite & self.flags['valid_hit'] & outgoing
            self.flags['landed'] |= first_landing
            self.flags['success'] |= success
            return_reward += 12.*success.float()-3.*(first_landing & ~success).float()
            self.flags['paddle_touch'] = paddle
            self.flags['table_touch'] = table
            # Once an env fails during this control step, freeze its target at
            # the actual configuration for remaining substeps; no state teleport.
            target = torch.where(unsafe[:, None], d.joint_pos.clone(), target)
        e.episode_length_buf += 1
        e.ball_episode_length_buf += 1
        h, tilt, drift, foot, load = self.body_values()
        body_fail = (h < float(self.cfg.reference_base_height)-.18) | (tilt > .45) | (drift > .5)
        unsafe |= body_fail
        ball_end = ((e.ball.data.root_pos_w-e.scene.env_origins)[:, 2] < .1) | self.flags['landed']
        terminated = unsafe | (ball_end & (not self.acceptance))
        time_out = (e.episode_length_buf >= e.max_episode_length) & ~terminated
        done = terminated | time_out
        self.flags['success'] &= ~unsafe
        # Dense shaping helps discover contacts, but never increments contact or
        # return counters. Event bonuses require measured filtered impulses.
        distance = torch.linalg.vector_norm(e.ball.data.root_pos_w-e.paddle_touch_point, dim=-1)
        reach = torch.exp(-distance.square()/.08)*(~self.flags['hit']).float()
        safe_penalty = ((self.safe_distance-min_margins).clamp_min(0.)/self.safe_distance).square().sum(-1)
        equilibrium = torch.exp(-((h-float(self.cfg.reference_base_height))/.05).square()-tilt.square()/.01-drift.square()/.01)
        reward = self.step_dt*(.2*reach+.5*equilibrium-4.*safe_penalty-.02*action_cost-2.*self.filtered-(load-.8).clamp_min(0.).square())
        reward += contact_reward+direction_reward+return_reward
        reward = torch.where(unsafe, reward.clamp_max(0.)-10., reward)
        assert torch.isfinite(reward).all(), 'NONFINITE TRAINING REWARD'
        # Capture all terminal evidence before reset; timeout is distinct from failure.
        evidence = dict(height=h.clone(), tilt=tilt.clone(), drift=drift.clone(), feet=foot.clone(), load=load.clone(),
                        min_margins=min_margins.clone(), body_peak=body_peak.clone(), paddle_peak=ball_peak.clone(),
                        guard_failure=unsafe.clone(), no_path=early.clone(), timeout=time_out.clone(),
                        episode_step=e.episode_length_buf.clone(), actual_q=d.joint_pos.clone(),
                        target_q=d.joint_pos_target.clone(), velocity=d.joint_vel.clone(),
                        ball_position_env=(e.ball.data.root_pos_w-e.scene.env_origins).clone(),
                        ball_velocity=e.ball.data.root_lin_vel_w.clone(), paddle_normal=e.paddle_normal.clone(),
                        hit=self.flags['hit'].clone(), success=self.flags['success'].clone())
        evidence.update({key: value.detach().clone() for key, value in self.guard_info.items()})
        self.total['episodes'] += int(done.sum())
        self.total['failures'] += int((done & unsafe).sum())
        self.total['guard_stops'] += int((done & (unsafe & ~body_fail)).sum())
        self.total['contacts'] += int(((contact_reward > 0) & ~unsafe).sum())
        self.total['returns'] += int(((return_reward > 0) & ~unsafe).sum())
        self.last_evidence = evidence
        # Append pre-reset transitions, then seed only reset rows with fresh state.
        self._advance_histories()
        ids = done.nonzero(as_tuple=False).flatten()
        self.reset(ids)
        obs, _ = self.get_observations()
        self.extras = {'observations': {'critic': obs}, 'time_outs': time_out,
                       'log': {'ready/contacts': self.total['contacts'], 'ready/returns': self.total['returns'],
                               'ready/guard_stops': self.total['guard_stops']}}
        if not e.headless:
            e.sim.render()
        return obs, reward, done, self.extras

    def verify_frozen(self):
        e = self.e
        assert torch.equal(e.robot.data.default_joint_pos, self.initial_defaults)
        assert torch.equal(e._h1_2_reference_paddle_offset_b, self.initial_ref)
        assert torch.equal(e.robot.root_physx_view.get_dof_stiffnesses(), self.initial_kp)
        assert torch.equal(e.robot.root_physx_view.get_dof_dampings(), self.initial_kd)
        for module, state in zip(self.frozen_modules, self.frozen_states):
            assert all(torch.equal(v.cpu(), state[k]) for k, v in module.state_dict().items()), 'FROZEN STANDING STATE CHANGED'
        assert sha(RUN/'model_499.pt') == STAND_SHA

    def close(self):
        self.e.close()
