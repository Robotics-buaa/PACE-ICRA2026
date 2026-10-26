"""Full-body PACE environment with ready resets and measured collision events."""
from copy import deepcopy
import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import RewardTermCfg
from legged_lab.envs.h1_2_tt.h1_2_tt_env import H1_2TTEnv
from legged_lab.envs.h1_2_tt.h1_2_tt_config import H1_2TableTennisEnvCfg, H1_2TableTennisAgentCfg
from legged_lab.assets.unitree.h1_2_tt import H1_2_CONTROL_JOINTS
from ready_pose import RIGHT_READY_RAD
from runtime import configure_self_contacts
from physical_events import PhysicalEvents
from pace_intermediate import compute_intermediate_values
from pace_safety import Safety
import pace_common as common
import pace_rewards as rew


def make_configs():
    cfg, agent = deepcopy(H1_2TableTennisEnvCfg()), deepcopy(H1_2TableTennisAgentCfg())
    cfg.ready_base_xy = (-1.8, 0.)
    cfg.scene.robot.init_state.pos = (*cfg.ready_base_xy, cfg.scene.robot.init_state.pos[2])
    # Root XY/yaw randomness, noise, delays, pushes, masses, material and serves
    # retain PACE settings. Only joint/reset velocity randomization conflicts
    # with the required exact preparation reset.
    for name in ('reset_locomotion_joints', 'reset_manipulation_joints'):
        event = getattr(cfg.domain_rand.events, name)
        event.params['velocity_range'] = (0., 0.)
    cfg.scene.ball.spawn.activate_contact_sensors = True
    cfg.reward.joint_deviation_right_arms.func = rew.ready_arm_deviation
    cfg.reward.reward_table_success.func = rew.actual_return
    cfg.reward.early_body_clearance = RewardTermCfg(func=rew.early_margin, weight=-4.)
    cfg.reward.invalid_physical_hit = RewardTermCfg(func=rew.invalid_contact, weight=-150.)
    cfg.reward.supported_height = RewardTermCfg(func=rew.body_height, weight=4.)
    common.BINDINGS[id(cfg)] = configure_self_contacts(cfg, H1_2_CONTROL_JOINTS)
    agent.experiment_name = 'h1_2_table_tennis_ready_pace'
    agent.run_name = 'h1_2_ready_pace_pred_full'
    agent.resume = False
    assert cfg.robot.num_actions == 27
    return cfg, agent


class PaceReadyEnv(H1_2TTEnv):
    def __init__(self, cfg, headless):
        self.safety = None
        super().__init__(cfg, headless)
        fk, audit = common.BINDINGS.pop(id(cfg))
        self.safety = Safety(self, fk, audit)
        self.reset(torch.arange(self.num_envs, device=self.device))
        self.scene.write_data_to_sim()
        common.ENV = self
        print('[PACE-READY] Full 27-joint policy; original future body targets; no stance override.', flush=True)

    def init_buffers(self):
        super().init_buffers()
        self.ready_q = self.robot.data.default_joint_pos.clone()
        for name, value in RIGHT_READY_RAD.items():
            self.ready_q[:, self.robot.joint_names.index(name)] = value
        self.action_center = self.ready_q[:, self.action_joint_ids].clone()
        self.serve_epoch = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.events = PhysicalEvents(self.num_envs, torch,
            lambda a: torch.as_tensor(a, device=self.device), self.cfg.table)
        self._actor_history = torch.zeros(self.num_envs, 5, 99, device=self.device)
        self._critic_history = torch.zeros(self.num_envs, 5, 116, device=self.device)
        self._raw_action = torch.zeros(self.num_envs, 27, device=self.device)
        self._prediction_noise = torch.zeros(self.num_envs,5,device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._total_gpu = {k: torch.zeros((),dtype=torch.long,device=self.device) for k in ('episodes','failures','guard_stops','serves','raw_contacts','contacts','returns','no_path','forecast_rejections','physical_body_contact','physical_clearance')}
        self.last_evidence = {}
        self.predictor_sample = None

    @property
    def total(self):
        values = torch.stack(list(self._total_gpu.values())).cpu().tolist()
        return dict(zip(self._total_gpu,values))

    def compute_paddle_touch(self):
        # Keep the verified H1-2 geometry; do not manufacture touch from distance.
        self.ball_global_pos = self.ball.data.root_pos_w
        q = self.robot.data.body_quat_w[:, self.paddle_body_id]
        self.paddle_quat = q/q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.paddle_touch_point = self.robot.data.body_pos_w[:, self.paddle_body_id]+math_utils.quat_apply(self.paddle_quat, self._paddle_center_offset)
        self.paddle_normal = math_utils.quat_apply(self.paddle_quat, self._paddle_local_normal)
        self.paddle_pos = self.paddle_touch_point-self.scene.env_origins
        self.paddel_ball_distance = torch.linalg.vector_norm(self.ball_global_pos-self.paddle_touch_point, dim=1)-self._ball_radius
        self.ball_contact = torch.zeros(self.num_envs, device=self.device)

    def reset_ball(self, env_ids):
        super().reset_ball(env_ids)
        if hasattr(self, 'events') and len(env_ids):
            self.events.reset(env_ids)
            self.serve_epoch[env_ids] += 1
            self.ball_prediction[env_ids] = 0.

    def reset(self, env_ids):
        if not len(env_ids):
            return
        super().reset(env_ids)
        if not hasattr(self, 'ready_q'):
            return
        q = self.ready_q[env_ids]
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        self.robot.set_joint_velocity_target(torch.zeros_like(q), env_ids=env_ids)
        self.robot.set_joint_effort_target(torch.zeros_like(q), env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids)
        self._raw_action[env_ids] = 0.
        self._prediction_noise[env_ids] = 0.
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.compute_perception()
        self.compute_paddle_touch()
        if not hasattr(self,'_pace_ready_offset_b'):
            self._pace_ready_offset_b = math_utils.quat_rotate_inverse(
                self.robot.data.root_link_quat_w,
                self.paddle_touch_point-self.robot.data.root_link_pos_w).detach().clone()
        self.delayed_perception[env_ids] = self.current_perception[env_ids]
        self.sync_events()
        self.compute_intermediate_values()
        actor, critic = self.compute_current_observations_perception()
        assert actor.shape[1] == 99 and critic.shape[1] == 116
        self._actor_history[env_ids] = actor[env_ids,None,:]
        self._critic_history[env_ids] = critic[env_ids,None,:]
        if self.safety is not None:
            self.safety.reset(env_ids)
            self.scene.write_data_to_sim()

    def sync_events(self):
        f, p = self.events.flags, self.events.pulses
        self.has_touch_own_table_prev = f['own'].clone()
        self.has_touch_own_table = f['own'].clone()
        self.has_touch_paddle = f['valid'].clone()
        self.has_touch_opponent_table_just_now = p['return'].clone()
        self.has_touch_opo_table_prev = f['success'].clone()
        self.has_return_own_table2_prev = f['own_return'].clone()
        self.ball_contact_rew = p['hit'].float()

    compute_intermediate_values = compute_intermediate_values

    def _h1_2_paddle_offset_xy(self):
        if not hasattr(self,'_pace_ready_offset_b'):
            return super()._h1_2_paddle_offset_xy()
        c,s = torch.cos(self.robot.data.heading_w),torch.sin(self.robot.data.heading_w)
        offset = self._pace_ready_offset_b
        return torch.stack((c*offset[:,0]-s*offset[:,1],s*offset[:,0]+c*offset[:,1]),-1)

    def get_observations(self):
        actor = self._actor_history.reshape(self.num_envs, -1).clamp(-self.clip_obs, self.clip_obs)
        critic = self._critic_history.reshape(self.num_envs, -1).clamp(-self.clip_obs, self.clip_obs)
        self.extras['observations'] = {'critic': critic}
        return actor, self.extras

    def refresh_prediction_observations(self):
        # Refresh only learned-prediction fields, with no extra history append or
        # additional noise sampling. Actor never receives analytic future labels.
        robot_pos = self.robot.data.root_link_pos_w-self.table.data.root_link_pos_w
        target = self._h1_2_base_target(self.ball_prediction)
        prediction = torch.cat((self.ball_prediction*self.obs_scales.ball_pos,
            (target[:,:2]-robot_pos[:,:2])*self.obs_scales.robot_pos),-1)
        self._actor_history[:, -1, 93:98] = prediction+self._prediction_noise
        return self.get_observations()

    def append_observations(self, reset_ids):
        actor, critic = self.compute_current_observations_perception()
        clean_prediction = actor[:,93:98].clone()
        if self.add_noise:
            actor += (2*torch.rand_like(actor)-1)*self.noise_scale_vec
        self._prediction_noise = actor[:,93:98]-clean_prediction
        self._actor_history = torch.cat((self._actor_history[:,1:], actor[:,None,:]),1)
        self._critic_history = torch.cat((self._critic_history[:,1:], critic[:,None,:]),1)
        # Resets must seed all five slots, never retain one terminal frame.
        self._actor_history[reset_ids] = actor[reset_ids,None,:]
        self._critic_history[reset_ids] = critic[reset_ids,None,:]

    def check_reset_isolation(self):
        ids = torch.arange(0,self.num_envs,2,device=self.device)
        other = torch.arange(1,self.num_envs,2,device=self.device)
        def state():
            d = self.robot.data
            return dict(q=d.joint_pos,qd=d.joint_vel,target=d.joint_pos_target,
                effort=d.joint_effort_target,root=d.root_state_w,ball=self.ball.data.root_state_w,
                actor=self._actor_history,critic=self._critic_history,
                epoch=self.serve_epoch,raw=self._raw_action,perception=self.delayed_perception)
        kept = {k:v[other].clone() for k,v in state().items()}
        self._actor_history[ids] = 7.
        self._critic_history[ids] = 8.
        self._raw_action[ids] = .3
        self.robot.write_joint_state_to_sim(self.ready_q[ids],torch.full_like(self.ready_q[ids],.1),env_ids=ids)
        self.reset(ids)
        for k,v in state().items():
            assert torch.equal(v[other],kept[k]), 'SPARSE RESET CONTAMINATION: '+k
        d = self.robot.data
        assert torch.allclose(d.joint_pos[ids],self.ready_q[ids],atol=1e-6)
        assert torch.allclose(d.joint_pos_target[ids],self.ready_q[ids],atol=1e-6)
        assert (d.joint_vel[ids].abs() < 1e-6).all() and (d.joint_vel_target[ids].abs() < 1e-6).all()
        assert (self._raw_action[ids] == 0).all()
        assert torch.equal(self._actor_history[ids],self._actor_history[ids,:1].expand(-1,5,-1))
        print('[PACE-READY] NEW full-body reset/target/history isolation: PASS',flush=True)
        self.reset(torch.arange(self.num_envs,device=self.device))

    def update_prediction(self, preds):
        assert preds.shape == (self.num_envs,3) and torch.isfinite(preds).all()
        self.ball_prediction = preds.detach().clone()
        self.ball_prediction_vis = self.ball_prediction+self.scene.env_origins

    def check_reset(self):
        original, timeout = super().check_reset()
        d = self.robot.data
        height = d.root_link_pos_w[:,2]-self.scene.env_origins[:,2]
        tilt = torch.acos((-d.projected_gravity_b[:,2]).clamp(-1.,1.))
        healthy = (height >= float(self.cfg.reference_base_height)-.18) & (tilt <= .45)
        workspace = ((self.robot_pos[:,0] >= -3.6) & (self.robot_pos[:,0] <= -1.35)
                     & (self.robot_pos[:,1].abs() <= 1.1))
        # Original PACE table workspace bounds apply; no initial-XY anchor.
        failed = (original & ~timeout) | ~healthy | ~workspace | self._unsafe
        return failed | timeout, timeout & ~failed

    @torch.no_grad()
    def step(self, actions):
        first_step=self.predictor_sample is None
        if first_step:
            common.phase('FIRST_ENV_STEP_BEGIN')
        assert actions.shape == (self.num_envs,27) and torch.isfinite(actions).all()
        self._raw_action = actions.detach().clone()
        delayed = self.action_buffer.compute(actions)
        proposed = self.robot.data.joint_pos.clone()
        proposed[:,self.action_joint_ids] = delayed.clamp(-self.clip_actions,self.clip_actions)*self.action_scale+self.action_center
        target, unsafe = self.safety.select(proposed)
        self.events.begin_step()
        body_peak = actions.new_zeros(self.num_envs)
        paddle_peak = actions.new_zeros(self.num_envs)
        epoch = self.serve_epoch.clone()
        for _ in range(self.cfg.sim.decimation):
            previous_v = self.ball.data.root_lin_vel_w.clone()
            target = torch.where(unsafe[:,None], self.robot.data.joint_pos.clone(), target)
            self.robot.set_joint_position_target(target)
            self.safety.gravity.apply(target)
            self.aero.apply_to_rigid_object(self.ball)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            self.sim_step_counter += 1
            self.compute_perception()
            self.compute_paddle_touch()
            body_force, forces = self.safety.contacts.read()
            assert torch.isfinite(body_force).all() and torch.isfinite(forces).all()
            margins = self.safety.actual()
            self.safety.minimum = torch.minimum(self.safety.minimum,margins)
            unsafe |= (body_force > .1) | (margins < self.safety.hard).any(-1) | ~torch.isfinite(margins).all(-1)
            body_peak = torch.maximum(body_peak,body_force)
            paddle_peak = torch.maximum(paddle_peak,forces[:,0])
            self.events.update(self.ball_pos, previous_v, self.ball.data.root_lin_vel_w,
                               forces, self.safety.contacts.table_force_z, unsafe)
        self.safety.last = target.clone()
        self._unsafe = unsafe
        self.episode_length_buf += 1
        self.ball_episode_length_buf += 1
        self.command_generator.compute(self.step_dt)
        if 'interval' in self.event_manager.available_modes:
            self.event_manager.apply(mode='interval',dt=self.step_dt)
        self.sync_events()
        self.compute_intermediate_values()
        self.reset_buf, self.time_out_buf = self.check_reset()
        failed = self.reset_buf & ~self.time_out_buf
        self.events.suppress_unsafe(failed)
        self.sync_events()
        self.curriculum_manager.compute()
        # All rewards/events are settled BEFORE any robot or ball reset.
        reward = self.reward_manager.compute(self.step_dt)
        assert torch.isfinite(reward).all()
        ball_end = (self.ball.data.root_pos_w[:,2] < .1) | (self.ball_episode_length_buf >= self.max_ball_episode_length)
        end = ball_end | self.reset_buf
        self._total_gpu['episodes'] += self.reset_buf.sum()
        self._total_gpu['failures'] += failed.sum()
        self._total_gpu['guard_stops'] += unsafe.sum()
        self._total_gpu['serves'] += end.sum()
        for key,pulse in [('raw_contacts','raw_hit'),('contacts','hit'),('returns','return')]:
            self._total_gpu[key] += self.events.pulses[pulse].sum()
        self._total_gpu['no_path'] += self.safety.no_path.sum()
        self._total_gpu['forecast_rejections'] += (~self.safety.info['guard_forecast_ok']).sum()
        self._total_gpu['physical_body_contact'] += (body_peak > .1).sum()
        self._total_gpu['physical_clearance'] += (self.safety.minimum < self.safety.hard).any(-1).sum()
        self.predictor_sample = dict(position=self.ball_pos.clone(), target=self.ball_future_pose.clone(),
            epoch=epoch, valid=~self.mask_invalid & ~failed)
        transitions = {k:v.clone() for k,v in self.events.pulses.items()}
        transitions['serve_end'] = end.clone()
        self.last_evidence = dict(episode_step=self.episode_length_buf.clone(), guard_failure=failed.clone(),
            no_path=self.safety.no_path.clone(), min_margins=self.safety.minimum.clone(), body_peak=body_peak,
            paddle_peak=paddle_peak, height=self.robot.data.root_link_pos_w[:,2]-self.scene.env_origins[:,2],
            tilt=torch.acos((-self.robot.data.projected_gravity_b[:,2]).clamp(-1.,1.)),
            robot_position_table=self.robot_pos.clone(),
            actual_q=self.robot.data.joint_pos.clone(), target_q=self.robot.data.joint_pos_target.clone(),
            velocity=self.robot.data.joint_vel.clone(), ball_position_env=self.ball_pos.clone(),
            ball_velocity=self.ball.data.root_lin_vel_w.clone(), **self.safety.info)
        done, timeout = self.reset_buf.clone(), self.time_out_buf.clone()
        ids = done.nonzero(as_tuple=False).flatten()
        ball_ids = (ball_end & ~done).nonzero(as_tuple=False).flatten()
        if len(ball_ids):
            self.reset_ball(ball_ids)
        self.reset(ids)
        self.ball_reset_ids = end.nonzero(as_tuple=False).flatten()
        self.compute_perception()
        self.compute_paddle_touch()
        self.sync_events()
        self.compute_intermediate_values()
        self.delayed_perception = self.perception_buffer.compute(self.current_perception)
        self.append_observations(ids)
        self.extras['time_outs'] = timeout
        self.extras['pace_events'] = transitions
        self.extras.setdefault('log',{}).update({'pace/'+k: v.detach().clone().float() for k,v in self._total_gpu.items()})
        if not self.headless:
            self.sim.render()
        obs,extras = self.get_observations()
        if first_step:
            common.phase('FIRST_ENV_STEP_END')
        return obs,reward,done,extras

    def close(self):
        if not getattr(self,'_is_closed',True):
            if self.safety is not None:
                runner = common.RUNNER
                if runner is not None and getattr(runner,'learn_completed',False):
                    runner.save(str(__import__('pathlib').Path(runner.log_dir)/'model_final.pt'))
                self.safety.verify()
                common.verify()
                common.CLOSED_VERIFIED = True
            super().close()
