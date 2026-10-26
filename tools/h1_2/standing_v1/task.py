"""Independent H1-2 standing stage. No mutation of original task/config/assets.

The TT 99/116 observation layouts and 27 actions are retained, with no-ball fields
neutralized. Equal dimensions do NOT make the old TT policy a trained stand policy.
"""
from copy import deepcopy

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from legged_lab.assets.unitree.h1_2_tt import H1_2_PADDLE_BODY_NAME, H1_2_PADDLE_CENTER_OFFSET
from legged_lab.envs.h1_2_tt.h1_2_tt_env import H1_2TTEnv
from legged_lab.envs.h1_2_tt.h1_2_tt_config import H1_2TableTennisEnvCfg, H1_2TableTennisAgentCfg


def height(env):
    return env.robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def height_gate(env):
    # Zero at reference-12cm, full at reference-4cm. A 0.80m crouch earns no gated positives.
    delta = height(env) - float(env.cfg.reference_base_height)
    return ((delta + .12) / .08).clamp(0., 1.)


def reward_height(env):
    return torch.exp(-((height(env) - float(env.cfg.reference_base_height)) / .05).square())


def reward_upright(env):
    error = env.robot.data.projected_gravity_b[:, :2].square().sum(dim=1)
    return height_gate(env) * torch.exp(-error / .01)


def reward_quiet(env):
    v = env.robot.data.root_lin_vel_w.square().sum(dim=1)
    w = env.robot.data.root_ang_vel_b.square().sum(dim=1)
    return height_gate(env) * torch.exp(-v / .01 - w / .09)


def reward_anchor(env):
    pos = env.robot.data.root_link_pos_w - env.scene.env_origins
    error = (pos[:, :2] - env.stand_goal_env()[:, :2]).square().sum(dim=1)
    return height_gate(env) * torch.exp(-error / .0025)


def reward_support(env):
    forces = env.contact_sensor.data.net_forces_w[:, env.feet_cfg.body_ids, 2]
    return height_gate(env) * (forces > 5.).float().mean(dim=1)


def cost_upper_pose(env):
    q = env.robot.data.joint_pos - env.robot.data.default_joint_pos
    return q[:, env.stand_upper_ids].square().mean(dim=1)


def cost_foot_motion(env):
    # Foot link velocity is a support-motion proxy, not a measured sole slip velocity.
    return env.robot.data.body_link_lin_vel_w[:, env.stand_foot_ids, :2].square().sum(dim=(1, 2))


def cost_effort_pressure(env):
    d = env.robot.data
    estimate = (d.joint_stiffness * (d.joint_pos_target - d.joint_pos)
                + d.joint_damping * (d.joint_vel_target - d.joint_vel) + d.joint_effort_target)
    ratio = estimate.abs() / d.joint_effort_limits.clamp_min(1e-6)
    # This is an estimate from implicit PD, not measured PhysX drive effort.
    return (ratio - .8).clamp_min(0.).square().sum(dim=1)


def cost_action_rate(env):
    return env.stand_action_delta.square().sum(dim=1)


def cost_projection(env):
    return env.stand_projection_delta.square().sum(dim=1)


def cost_joint_margin(env):
    q, limits = env.robot.data.joint_pos, env.robot.data.soft_joint_pos_limits
    return ((limits[:, :, 0] - q).clamp_min(0.) + (q - limits[:, :, 1]).clamp_min(0.)).square().sum(dim=1)


def cost_failure(env):
    return (env.reset_buf & ~env.time_out_buf).float()


@configclass
class StandingRewards:
    standing_height = RewardTermCfg(func=reward_height, weight=4.)
    standing_upright = RewardTermCfg(func=reward_upright, weight=3.)
    standing_quiet = RewardTermCfg(func=reward_quiet, weight=2.)
    standing_anchor = RewardTermCfg(func=reward_anchor, weight=2.)
    standing_support = RewardTermCfg(func=reward_support, weight=1.)
    upper_ready_pose = RewardTermCfg(func=cost_upper_pose, weight=-.5)
    foot_motion = RewardTermCfg(func=cost_foot_motion, weight=-2.)
    estimated_effort_pressure = RewardTermCfg(func=cost_effort_pressure, weight=-2.)
    applied_action_rate = RewardTermCfg(func=cost_action_rate, weight=-.05)
    impossible_target_request = RewardTermCfg(func=cost_projection, weight=-1.)
    joint_margin = RewardTermCfg(func=cost_joint_margin, weight=-2.)
    standing_failure = RewardTermCfg(func=cost_failure, weight=-50.)


def make_cfg():
    # Construct the original class fully first: its __post_init__ expects original reward fields.
    cfg = deepcopy(H1_2TableTennisEnvCfg())
    cfg.reward = StandingRewards()
    cfg.curriculum = {}
    cfg.ready_base_xy = (-1.8, 0.)
    cfg.scene.robot.init_state.pos = (*cfg.ready_base_xy, cfg.scene.robot.init_state.pos[2])
    cfg.scene.max_episode_length_s = 20.
    cfg.scene.terrain_type = 'plane'
    cfg.scene.terrain_generator = None
    cfg.scene.num_envs = 2
    cfg.scene.seed = 42
    cfg.noise.add_noise = False
    cfg.domain_rand.action_delay.enable = False
    cfg.domain_rand.perception_delay.enable = False
    cfg.domain_rand.events.push_robot = None
    cfg.domain_rand.events.add_base_mass = None
    cfg.domain_rand.events.physics_material = None
    cfg.domain_rand.events.reset_base.params['pose_range'] = {k: (0., 0.) for k in ('x', 'y', 'yaw')}
    cfg.domain_rand.events.reset_base.params['velocity_range'] = {k: (0., 0.) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')}
    cfg.domain_rand.events.reset_locomotion_joints.params['position_range'] = (1., 1.)
    cfg.domain_rand.events.reset_manipulation_joints.params['position_range'] = (0., 0.)
    for event in (cfg.domain_rand.events.reset_locomotion_joints, cfg.domain_rand.events.reset_manipulation_joints):
        if 'velocity_range' in event.params:
            event.params['velocity_range'] = (0., 0.)
    return cfg


def make_agent_cfg():
    cfg = deepcopy(H1_2TableTennisAgentCfg())
    cfg.experiment_name = 'h1_2_standing_v1'
    cfg.run_name = 'stand_v1'
    cfg.resume = False
    return cfg


class H1_2StandV1Env(H1_2TTEnv):
    def init_buffers(self):
        # Parent init_buffers calls virtual observation methods before it creates
        # paddle_pos/paddle_body_id. Resolve only robot-backed geometry beforehand.
        ids, names = self.robot.find_bodies(H1_2_PADDLE_BODY_NAME)
        assert names == [H1_2_PADDLE_BODY_NAME], 'STANDING PADDLE BODY MAPPING CHANGED'
        self._stand_paddle_body_id = ids[0]
        self._stand_paddle_center_offset = self.robot.data.body_pos_w.new_tensor(
            H1_2_PADDLE_CENTER_OFFSET).expand(self.num_envs, -1)
        super().init_buffers()
        self.stand_foot_ids = self.robot.find_bodies(
            ['left_ankle_roll_link', 'right_ankle_roll_link'], preserve_order=True)[0]
        self.stand_upper_ids = [j for j, n in enumerate(self.robot.joint_names)
                                if any(x in n for x in ('shoulder', 'elbow', 'wrist', 'torso'))]
        assert len(self.stand_foot_ids) == 2 and self.stand_upper_ids, 'STANDING BODY/JOINT MAPPING CHANGED'
        assert len(self.action_joint_ids) == 27, 'STANDING ACTION MAPPING CHANGED'
        shape = (self.num_envs, 27)
        self.stand_applied_actions = torch.zeros(shape, device=self.device)
        self.stand_action_delta = torch.zeros(shape, device=self.device)
        self.stand_projection_delta = torch.zeros(shape, device=self.device)
        self.stand_requested_targets = torch.zeros(shape, device=self.device)

    def stand_goal_env(self):
        return self.robot.data.default_root_state.new_tensor(
            [*self.cfg.ready_base_xy, float(self.cfg.reference_base_height)]).expand(self.num_envs, -1)

    def stand_goal_table(self):
        return self.stand_goal_env() - (self.table.data.root_link_pos_w - self.scene.env_origins)

    def stand_paddle_goal_env(self):
        return self.stand_goal_env() + self._h1_2_reference_paddle_offset_b

    def stand_paddle_position_env(self):
        # Match H1_2TTEnv.compute_paddle_touch without requiring its late buffers
        # or triggering hit bookkeeping while observation buffers are initialized.
        quat = self.robot.data.body_quat_w[:, self._stand_paddle_body_id]
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        center = self.robot.data.body_pos_w[:, self._stand_paddle_body_id] + math_utils.quat_apply(
            quat, self._stand_paddle_center_offset)
        return center - self.scene.env_origins

    def hold_standing_targets(self):
        self.robot_future_pos = self.stand_goal_table()
        self.robot_future_vel = torch.zeros_like(self.robot_future_pos)
        self.ball_future_pose = self.stand_paddle_goal_env()
        self.ball_prediction = self.ball_future_pose.clone()
        self.ball_future_t = self.robot_future_pos.new_zeros((self.num_envs, 1))
        self.mask_invalid = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

    def _h1_2_relative_target(self, actor_obs):
        robot_table = self.robot.data.root_link_pos_w - self.table.data.root_link_pos_w
        actor_obs[:, -3:-1] = (self.stand_goal_table()[:, :2] - robot_table[:, :2]) * self.obs_scales.robot_pos
        return actor_obs

    def standing_observations(self, actor, critic):
        assert actor.shape[1] == 99 and critic.shape[1] == 116, 'UPSTREAM OBSERVATION LAYOUT CHANGED'
        # 27-action TT layouts verified from current source. Remove the parked ball from both networks.
        actor[:, 87:93] = 0.
        actor[:, 93:96] = self.stand_paddle_goal_env() * self.obs_scales.ball_pos
        self._h1_2_relative_target(actor)
        critic[:, 87:93] = 0.
        critic[:, 93:96] = self.stand_paddle_goal_env() * self.obs_scales.ball_pos
        critic[:, 96:99] = self.stand_paddle_position_env() * self.obs_scales.ball_pos
        robot_table = self.robot.data.root_link_pos_w - self.table.data.root_link_pos_w
        critic[:, 103:106] = (self.stand_goal_table() - robot_table) * self.obs_scales.robot_pos
        critic[:, 106:108] = 0.
        critic[:, 109:111] = 0.
        return actor, critic

    def compute_current_observations(self):
        self.hold_standing_targets()
        return self.standing_observations(*super().compute_current_observations())

    def compute_current_observations_perception(self):
        self.hold_standing_targets()
        return self.standing_observations(*super().compute_current_observations_perception())

    def compute_intermediate_values(self):
        super().compute_intermediate_values()
        self.hold_standing_targets()

    def reset_ball(self, env_ids):
        super().reset_ball(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(ids):
            state = self.ball.data.default_root_state[ids].clone()
            state[:, :3] = self.scene.env_origins[ids] + state.new_tensor([100., 100., 10000.])
            state[:, 7:] = 0.
            self.reset_ball_state_buf[ids] = state
            self.ball.write_root_pose_to_sim(state[:, :7], ids)
            self.ball.write_root_velocity_to_sim(state[:, 7:], ids)

    def reset(self, env_ids):
        super().reset(env_ids)
        if len(env_ids) and hasattr(self, 'stand_applied_actions'):
            for buffer in (self.stand_applied_actions, self.stand_action_delta, self.stand_projection_delta):
                buffer[env_ids] = 0.
            self.stand_requested_targets[env_ids] = self.robot.data.default_joint_pos[env_ids][:, self.action_joint_ids]
        self.hold_standing_targets()

    def step(self, actions):
        assert not self.cfg.domain_rand.action_delay.enable, 'STAND V1 EXPECTS NO ACTION DELAY'
        ids = self.action_joint_ids
        clipped = torch.clip(actions, -self.clip_actions, self.clip_actions)
        requested = clipped * self.action_scale + self.robot.data.default_joint_pos[:, ids]
        limits = self.robot.data.joint_pos_limits[:, ids]
        projected = torch.maximum(torch.minimum(requested, limits[:, :, 1]), limits[:, :, 0])
        applied = (projected - self.robot.data.default_joint_pos[:, ids]) / self.action_scale
        self.stand_requested_targets = requested.detach().clone()
        self.stand_action_delta = applied - self.stand_applied_actions
        self.stand_projection_delta = requested - projected
        self.stand_applied_actions = applied.detach().clone()
        return super().step(applied)

    def check_reset(self):
        # Ignore ball serve counters; stand has its own explicit failure/20s timeout criteria.
        d = self.robot.data
        h = height(self)
        tilt = torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1., 1.))
        xy = d.root_link_pos_w[:, :2] - self.scene.env_origins[:, :2]
        distance = torch.linalg.vector_norm(xy - self.stand_goal_env()[:, :2], dim=1)
        failed = (h < float(self.cfg.reference_base_height) - .18) | (tilt > .45) | (distance > .5)
        timeout = self.episode_length_buf >= self.max_episode_length
        return failed | timeout, timeout


_registered = False


def register_tasks():
    global _registered
    if _registered:
        return
    from legged_lab.utils import task_registry
    task_registry.register('h1_2_stand_v1', H1_2StandV1Env, make_cfg(), make_agent_cfg())
    task_registry.register('h1_2_stand_v1_eval', H1_2StandV1Env, make_cfg(), make_agent_cfg())
    _registered = True
