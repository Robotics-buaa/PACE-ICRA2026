# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

"""H1_2-specific robot and paddle interfaces for the existing PACE environment."""

import torch
import isaaclab.utils.math as math_utils
from isaaclab.utils.buffers import CircularBuffer

from legged_lab.assets.unitree.h1_2_tt import (
    H1_2_PADDLE_BODY_NAME,
    H1_2_PADDLE_CENTER_OFFSET,
    H1_2_PADDLE_NORMAL,
    H1_2_CONTROL_JOINTS,
    H1_2_PADDLE_PARENT_BODY_NAME,
)
from legged_lab.envs.base.tt_env import TTEnv


class H1_2TTEnv(TTEnv):
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
        self._calibrate_h1_2_base_height()
        self._initialize_h1_2_geometry()
        super().init_buffers()
        ids, names = self.robot.find_bodies(H1_2_PADDLE_BODY_NAME)
        if names != [H1_2_PADDLE_BODY_NAME]:
            raise RuntimeError(f"Expected one H1-2 paddle body, found {names}")
        self.paddle_body_id = ids[0]
        if self.robot.num_joints != 27 or self.action_joint_names != H1_2_CONTROL_JOINTS:
            raise RuntimeError("H1-2 joint/action mapping mismatch")
        if self.obs_joint_names != self.action_joint_names:
            raise RuntimeError("H1-2 observation/action order mismatch")
        self.held_joint_ids = []
        self.wrist_joint_ids, self.wrist_joint_names = self.robot.find_joints(".*_wrist_.*_joint")
        if len(self.wrist_joint_ids) != 6:
            raise RuntimeError("Expected six native H1-2 wrist joints")
        self._paddle_center_offset = torch.tensor(H1_2_PADDLE_CENTER_OFFSET, device=self.device, dtype=torch.float32).expand(self.num_envs, -1)
        self._paddle_local_normal = torch.tensor(H1_2_PADDLE_NORMAL, device=self.device, dtype=torch.float32).expand(self.num_envs, -1)
        self._ball_radius = float(self.cfg.scene.ball.spawn.radius)


    def reset(self, env_ids):
        super().reset(env_ids)
        if len(env_ids) == 0:
            return
        ids = self.wrist_joint_ids
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids][:, ids],
            self.robot.data.default_joint_vel[env_ids][:, ids],
            joint_ids=ids, env_ids=env_ids,
        )
        # Hold the actual randomized reset pose; subsequent actions use the normal default offset.
        self.robot.set_joint_position_target(self.robot.data.joint_pos[env_ids].clone(), env_ids=env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.compute_perception()
        self.compute_paddle_touch()


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

    def _initialize_h1_2_geometry(self):
        """Measure the default paddle offset before randomized episode resets."""
        ids, names = self.robot.find_bodies(H1_2_PADDLE_BODY_NAME)
        if names != [H1_2_PADDLE_BODY_NAME]:
            raise RuntimeError(f"Expected one H1_2 paddle body, found {names}")
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
            H1_2_PADDLE_CENTER_OFFSET, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        center = self.robot.data.body_pos_w[:, ids[0]] + math_utils.quat_apply(
            paddle_quat, offset
        )
        measured = math_utils.quat_rotate_inverse(
            self.robot.data.root_link_quat_w,
            center - self.robot.data.root_link_pos_w,
        )
        if not bool(torch.isfinite(measured).all()):
            raise RuntimeError("Non-finite H1_2 reference paddle geometry")
        self._h1_2_measured_paddle_offset_b = measured.detach().clone()
        configured = self.cfg.reference_paddle_offset_b
        if configured is None:
            self._h1_2_reference_paddle_offset_b = measured.detach().clone()
        else:
            self._h1_2_reference_paddle_offset_b = torch.tensor(
                configured, device=self.device, dtype=torch.float32
            ).expand(self.num_envs, -1)

    def _h1_2_paddle_offset_xy(self):
        # Rotate the neutral-pose reference with the current heading.
        heading = self.robot.data.heading_w
        offset = self._h1_2_reference_paddle_offset_b
        c, s = torch.cos(heading), torch.sin(heading)
        return torch.stack(
            [c * offset[:, 0] - s * offset[:, 1],
             s * offset[:, 0] + c * offset[:, 1]], dim=-1,
        )

    def _h1_2_base_target(self, ball_target):
        """Convert an environment-frame ball target to a table-frame base target."""
        target = ball_target.clone()
        target[:, :2] -= self._h1_2_paddle_offset_xy()
        target[:, 2] = self.cfg.reference_base_height
        table_origin = self.table.data.root_link_pos_w - self.scene.env_origins
        return target - table_origin

    def _h1_2_relative_target(self, actor_obs):
        robot_pos = self.robot.data.root_link_pos_w - self.table.data.root_link_pos_w
        target = self._h1_2_base_target(self.ball_prediction)
        actor_obs[:, -3:-1] = (
            target[:, :2] - robot_pos[:, :2]
        ) * self.obs_scales.robot_pos
        return actor_obs

    def compute_current_observations(self):
        actor_obs, critic_obs = super().compute_current_observations()
        return self._h1_2_relative_target(actor_obs), critic_obs

    def compute_current_observations_perception(self):
        actor_obs, critic_obs = super().compute_current_observations_perception()
        return self._h1_2_relative_target(actor_obs), critic_obs

    def compute_intermediate_values(self):
        # Retain PACE ball dynamics and hit bookkeeping, then replace geometry targets.
        super().compute_intermediate_values()
        self.pos_pred_before_ro = self._h1_2_base_target(self.pos_pred_before)
        self.pos_pred_after_ro = self._h1_2_base_target(self.pos_pred_after)
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
        # Invalid predictions fall back to the actual H1_2 paddle center.
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

    def _calibrate_h1_2_base_height(self):
        """Align the neutral-pose foot collision bounds with the plane before any episode reset."""
        self._h1_2_reference_geometry = {}
        if not self.cfg.auto_reference_base_height:
            return
        from pxr import Usd, UsdGeom, UsdPhysics
        import omni.usd
        root = self.robot.data.default_root_state.clone()
        root[:, :3] += self.scene.env_origins
        self.robot.write_root_pose_to_sim(root[:, :7])
        self.robot.write_root_velocity_to_sim(root[:, 7:])
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)
        self.robot.set_joint_position_target(self.robot.data.default_joint_pos)
        self.sim.forward()
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage unavailable for height calibration")
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], False, True)
        lowest = []
        shapes = []
        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            body = stage.GetPrimAtPath(f"{self.scene.env_prim_paths[0]}/Robot/{name}")
            if not body.IsValid():
                raise RuntimeError(f"Foot prim missing for height calibration: {name}")
            bid = self.robot.find_bodies(name)[0][0]
            foot_pos = self.robot.data.body_link_pos_w[0, bid]
            foot_quat = self.robot.data.body_link_quat_w[0, bid]
            count = 0
            for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                if enabled is False:
                    continue
                extent = bounds.ComputeRelativeBound(prim, body).ComputeAlignedRange()
                if extent.IsEmpty():
                    continue
                lo, hi = extent.GetMin(), extent.GetMax()
                corners = torch.tensor([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])], device=self.device, dtype=torch.float32)
                world = foot_pos + math_utils.quat_apply(foot_quat.expand(len(corners), -1), corners)
                lowest.append((world[:, 2].min() - self.scene.env_origins[0, 2]).item())
                shapes.append(str(prim.GetPath()))
                count += 1
            if count == 0:
                raise RuntimeError(f"No foot collision geometry for height calibration: {name}")
        old_height = float(self.robot.data.default_root_state[0, 2].item())
        height = old_height - min(lowest) + float(self.cfg.initial_foot_clearance)
        if not 0.75 < height < 1.35:
            raise RuntimeError(f"Implausible H1-2 reference base height: {height}")
        self.cfg.reference_base_height = height
        old_pos = self.cfg.scene.robot.init_state.pos
        self.cfg.scene.robot.init_state.pos = (old_pos[0], old_pos[1], height)
        self.robot.data.default_root_state[:, 2] = height
        self._h1_2_reference_geometry.update({
            "base_height_before": old_height,
            "reference_base_height": height,
            "initial_foot_clearance": float(self.cfg.initial_foot_clearance),
            "foot_collision_prims": shapes,
            "method": "default-pose foot collision bounds on plane; not a learned standing stability test",
        })
        print(f"[H1-2] Reference base height from foot geometry: {height:.6f} m", flush=True)

