"""G1-specific robot and paddle interfaces for the existing PACE environment."""

import torch
import isaaclab.utils.math as math_utils
from isaaclab.utils.buffers import CircularBuffer

from legged_lab.assets.unitree.g1_tt import (
    G1_PADDLE_BODY_NAME,
    G1_PADDLE_CENTER_OFFSET,
    G1_PADDLE_NORMAL,
)
from legged_lab.envs.base.tt_env import TTEnv
from .g1_tt_config import G1_HAND_JOINTS


class G1TTEnv(TTEnv):
    def init_obs_buffer(self):
        if self.add_noise:
            return super().init_obs_buffer()
        # The base implementation only defines actor_obs when noise is enabled.
        actor_obs, _ = self.compute_current_observations()
        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length,
            batch_size=self.num_envs,
            device=self.device,
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length,
            batch_size=self.num_envs,
            device=self.device,
        )
        self.delayed_perception = actor_obs[..., -self.num_perception:]

    def init_buffers(self):
        self._initialize_g1_geometry()
        super().init_buffers()
        body_ids, names = self.robot.find_bodies(G1_PADDLE_BODY_NAME)
        if names != [G1_PADDLE_BODY_NAME]:
            raise RuntimeError(f"Expected one G1 paddle body, found {names}")
        self.paddle_body_id = body_ids[0]
        action_ids = set(self.action_joint_ids)
        self.held_joint_ids = [
            i for i in range(self.robot.num_joints) if i not in action_ids
        ]
        held_names = {self.robot.joint_names[i] for i in self.held_joint_ids}
        if held_names != set(G1_HAND_JOINTS):
            raise RuntimeError(f"Unexpected joints outside G1 actions: {held_names}")
        self._paddle_center_offset = torch.tensor(
            G1_PADDLE_CENTER_OFFSET, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        self._paddle_local_normal = torch.tensor(
            G1_PADDLE_NORMAL, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        self._ball_radius = float(self.cfg.scene.ball.spawn.radius)
        self._hold_hand_joints()

    def _hold_hand_joints(self):
        self.robot.set_joint_position_target(
            self.robot.data.default_joint_pos[:, self.held_joint_ids],
            joint_ids=self.held_joint_ids,
        )

    def reset(self, env_ids):
        super().reset(env_ids)
        if len(env_ids) == 0:
            return
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids][:, self.held_joint_ids],
            self.robot.data.default_joint_vel[env_ids][:, self.held_joint_ids],
            joint_ids=self.held_joint_ids,
            env_ids=env_ids,
        )
        self._hold_hand_joints()
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.compute_paddle_touch()

    def step(self, actions):
        self._hold_hand_joints()
        return super().step(actions)

    def compute_paddle_touch(self):
        self.ball_global_pos = self.ball.data.root_pos_w
        body_pos = self.robot.data.body_pos_w[:, self.paddle_body_id]
        body_quat = self.robot.data.body_quat_w[:, self.paddle_body_id]
        self.paddle_quat = body_quat / body_quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.paddle_touch_point = body_pos + math_utils.quat_apply(
            self.paddle_quat, self._paddle_center_offset
        )
        self.paddle_normal = math_utils.quat_apply(
            self.paddle_quat, self._paddle_local_normal
        )
        self.paddle_pos = self.paddle_touch_point - self.scene.env_origins

        # Preserve PACE's distance score and peak/retreat hit flag semantics.
        distance = torch.norm(self.ball_global_pos - self.paddle_touch_point, dim=1)
        self.paddel_ball_distance = distance - self._ball_radius
        contact_score = (
            self.cfg.ball.contact_threshold - self.paddel_ball_distance
        ) / self.cfg.ball.contact_threshold
        self.ball_contact = torch.clamp(contact_score, min=0.0, max=1.0)
        self.ball_contact_rew = torch.maximum(self.ball_contact_rew, self.ball_contact)
        new_hits = (contact_score > 0) & (self.ball_contact < self.ball_contact_rew)
        still_false = ~self.has_touch_paddle
        self.has_touch_paddle[still_false] = new_hits[still_false]

    def _initialize_g1_geometry(self):
        """Measure the default paddle offset before randomized episode resets."""
        ids, names = self.robot.find_bodies(G1_PADDLE_BODY_NAME)
        if names != [G1_PADDLE_BODY_NAME]:
            raise RuntimeError(f"Expected one G1 paddle body, found {names}")
        root = self.robot.data.default_root_state.clone()
        root[:, :3] += self.scene.env_origins
        self.robot.write_root_pose_to_sim(root[:, :7])
        self.robot.write_root_velocity_to_sim(root[:, 7:])
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos, self.robot.data.default_joint_vel
        )
        self.robot.set_joint_position_target(self.robot.data.default_joint_pos)
        self.sim.forward()
        paddle_quat = self.robot.data.body_quat_w[:, ids[0]]
        offset = torch.tensor(
            G1_PADDLE_CENTER_OFFSET, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        center = self.robot.data.body_pos_w[:, ids[0]] + math_utils.quat_apply(
            paddle_quat, offset
        )
        measured = math_utils.quat_rotate_inverse(
            self.robot.data.root_link_quat_w,
            center - self.robot.data.root_link_pos_w,
        )
        if not bool(torch.isfinite(measured).all()):
            raise RuntimeError("Non-finite G1 reference paddle geometry")
        self._g1_measured_paddle_offset_b = measured.detach().clone()
        configured = self.cfg.reference_paddle_offset_b
        if configured is None:
            self._g1_reference_paddle_offset_b = measured.detach().clone()
        else:
            self._g1_reference_paddle_offset_b = torch.tensor(
                configured, device=self.device, dtype=torch.float32
            ).expand(self.num_envs, -1)

    def _g1_paddle_offset_xy(self):
        # Rotate the neutral-pose reference with the current heading.
        heading = self.robot.data.heading_w
        offset = self._g1_reference_paddle_offset_b
        c, s = torch.cos(heading), torch.sin(heading)
        return torch.stack(
            [c * offset[:, 0] - s * offset[:, 1],
             s * offset[:, 0] + c * offset[:, 1]], dim=-1,
        )

    def _g1_base_target(self, ball_target):
        """Convert an environment-frame ball target to a table-frame base target."""
        target = ball_target.clone()
        target[:, :2] -= self._g1_paddle_offset_xy()
        target[:, 2] = self.cfg.reference_base_height
        table_origin = self.table.data.root_link_pos_w - self.scene.env_origins
        return target - table_origin

    def _g1_relative_target(self, actor_obs):
        robot_pos = self.robot.data.root_link_pos_w - self.table.data.root_link_pos_w
        target = self._g1_base_target(self.ball_prediction)
        actor_obs[:, -3:-1] = (
            target[:, :2] - robot_pos[:, :2]
        ) * self.obs_scales.robot_pos
        return actor_obs

    def compute_current_observations(self):
        actor_obs, critic_obs = super().compute_current_observations()
        return self._g1_relative_target(actor_obs), critic_obs

    def compute_current_observations_perception(self):
        actor_obs, critic_obs = super().compute_current_observations_perception()
        return self._g1_relative_target(actor_obs), critic_obs

    def compute_intermediate_values(self):
        # Retain PACE ball dynamics and hit bookkeeping, then replace geometry targets.
        super().compute_intermediate_values()
        self.pos_pred_before_ro = self._g1_base_target(self.pos_pred_before)
        self.pos_pred_after_ro = self._g1_base_target(self.pos_pred_after)
        mask_invalid = self.mask_invalid.unsqueeze(-1)
        self.robot_future_pos = torch.where(
            self.mask_before.unsqueeze(-1),
            self.pos_pred_before_ro, self.pos_pred_after_ro,
        )
        ready = self.robot_future_pos.new_tensor([
            *self.cfg.ready_base_xy, self.cfg.reference_base_height,
        ]).expand_as(self.robot_future_pos)
        table_origin = self.table.data.root_link_pos_w - self.scene.env_origins
        self.robot_future_pos = torch.where(
            mask_invalid, ready - table_origin, self.robot_future_pos
        )
        self.vel_ro_before = torch.clamp(
            (self.pos_pred_before_ro - self.robot_pos) * 4.0, min=-7.0, max=7.0
        )
        self.vel_ro_after = torch.clamp(
            (self.pos_pred_after_ro - self.robot_pos) * 4.0, min=-7.0, max=7.0
        )
        self.robot_future_vel = torch.where(
            self.mask_before.unsqueeze(-1), self.vel_ro_before, self.vel_ro_after
        )
        self.robot_future_vel = torch.where(
            mask_invalid, torch.zeros_like(self.robot_future_vel), self.robot_future_vel
        )
        # Invalid predictions fall back to the actual G1 paddle center.
        self.ball_future_pose = torch.where(
            mask_invalid, self.paddle_pos, self.ball_future_pose
        )
        landing_vis = torch.stack([
            self.predict_x_land, self.predict_y_land,
            torch.full_like(self.predict_x_land, 0.78),
        ], dim=-1)
        self.ball_future_pose_vis = torch.where(
            self.touched_paddel_no_bounce_table.unsqueeze(-1),
            landing_vis, self.ball_future_pose,
        ) + self.scene.env_origins

