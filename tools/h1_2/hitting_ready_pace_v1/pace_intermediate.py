# Derived from received PACE TTEnv/H1_2TTEnv (BSD-3-Clause).
# Original analytic targets retained; proximity event block replaced.
import torch

def compute_intermediate_values(self):
    # print("has_touch_paddle", self.has_touch_paddle)
    # self.ball_pos = self.ball.data.root_pos_w - self.table.data.root_pos_w  # Local (offset) position wrt table
    self.ball_quat = self.ball.data.root_quat_w
    self.ball_vel = self.ball.data.root_vel_w
    self.ball_linvel = self.ball.data.root_lin_vel_w
    self.ball_angvel = self.ball.data.root_ang_vel_w
    self.robot_linvel= self.robot.data.root_lin_vel_w
    self.ball_contact_rew = self.ball_contact_rew * (self.has_touch_paddle * ~self.has_touch_paddle_rew) # Mask if previously already gained reward
    self.ball_landing_dis_rew = self.has_touch_paddle & ~self.has_touch_paddle_rew # Set True if has_touch_paddle_rew is True, compute landing dis reward once, set False if previously True

    # Task-local physical contact flags are authoritative, never AABB scores.
    self.has_first_bounce_prev = self.has_first_bounce.clone()
    self.has_first_bounce = self.events.flags['own'].clone()
    self.touched_paddel_no_bounce_table = self.has_touch_paddle & ~(self.has_return_own_table2_prev | self.has_touch_opo_table_prev)
    self.paddle_pos = self.paddle_touch_point - self.scene.env_origins 
    has_bounced = self.has_touch_own_table_prev
    vz = self.ball_linvel[:, 2]
    z = self.ball_pos[:, 2]
    x = self.ball_pos[:, 0]
    y = self.ball_pos[:, 1]
    vx = self.ball_linvel[:, 0]
    vy = self.ball_linvel[:, 1]

    g=9.81
    body_height=0.69
    vel_max=7.0
    paddle_y_offset = -0.60

    self.mask_before = (has_bounced == 0).squeeze(-1)
    self.mask_after = (has_bounced == 1).squeeze(-1)

    h = 0.78
    D = vz.pow(2) + 2.0 * g * (z - h)
    sqrtD = torch.sqrt(torch.clamp(D, min=0.0))
    t_land = (vz + sqrtD) / g
    t_land = torch.clamp(t_land, min=0.0)
    self.valid_before = (D >= 0.0) & self.mask_before
    t1 = torch.where(self.valid_before, t_land, torch.zeros_like(t_land))
    vz_prime = vz - g * t1
    vz_dd = -0.9 * vz_prime
    # Drag-adjusted ascent after first bounce
    k_scalar = torch.tensor(max(1e-8, getattr(self, 'ball_drag_k', 0.13)), device=self.device, dtype=vz.dtype)
    sqrt_gk = torch.sqrt(torch.clamp(torch.tensor(g, device=self.device, dtype=vz.dtype) * k_scalar, min=1e-12))
    sqrt_kg = torch.sqrt(torch.clamp(k_scalar / torch.tensor(g, device=self.device, dtype=vz.dtype), min=0.0))
    vz_up = torch.clamp(vz_dd, min=0.0)
    t2_drag = torch.atan(torch.clamp(vz_up * sqrt_kg, min=0.0)) / torch.clamp(sqrt_gk, min=1e-12)
    delta_z_drag = (0.5 / k_scalar) * torch.log1p(torch.clamp((k_scalar / torch.tensor(g, device=self.device, dtype=vz.dtype)) * vz_up.pow(2), min=0.0))
    self.t_before = t1 + t2_drag
    zpb = delta_z_drag + h

    # Horizontal displacement under quadratic drag: s(T) = (1/k) * ln(1 + k |v0| T) * sign(v0)
    def horiz_disp(v0, T):
        vabs = torch.abs(v0)
        return (1.0 / k_scalar) * torch.sign(v0) * torch.log1p(torch.clamp(k_scalar * vabs * T, min=0.0))

    # Drag-adjusted ascent for current upward motion (case 'a')
    vz_up_a = torch.clamp(vz, min=0.0)
    t_after_drag = torch.atan(torch.clamp(vz_up_a * sqrt_kg, min=0.0)) / torch.clamp(sqrt_gk, min=1e-12)
    delta_z_drag_a = (0.5 / k_scalar) * torch.log1p(torch.clamp((k_scalar / torch.tensor(g, device=self.device, dtype=vz.dtype)) * vz_up_a.pow(2), min=0.0))
    self.t_after = t_after_drag
    zpa = z + delta_z_drag_a
    self.predict_land_t = t_land
    self.predict_x_land = x + horiz_disp(vx, t_land) # acounts for both landing on our side and the opponent's side of table
    self.predict_y_land = y + horiz_disp(vy, t_land)
    predict_ball_land_vis=torch.stack([self.predict_x_land, self.predict_y_land, torch.ones_like(x)*h], dim=-1)


    dx_ascent = horiz_disp(0.7 *vx, t2_drag)
    dy_ascent = horiz_disp(0.7 *vy, t2_drag)
    xpb = self.predict_x_land + dx_ascent
    ypb = self.predict_y_land + dy_ascent
    dx_after = horiz_disp(vx, t_after_drag)
    dy_after = horiz_disp(vy, t_after_drag)
    xpa = x + dx_after
    ypa = y + dy_after

    xpb=torch.clamp(xpb, max=-1.6)
    xpa=torch.clamp(xpa, max=-1.6)

    self.pos_pred_before = torch.stack([xpb, ypb, zpb], dim=-1)
    self.pos_pred_after = torch.stack([xpa, ypa, zpa], dim=-1)

    self.pos_pred_before_ro = torch.stack([xpb - 0.1, ypb - paddle_y_offset, torch.ones_like(xpb) * body_height], dim=-1)
    self.pos_pred_after_ro = torch.stack([xpa - 0.1, ypa - paddle_y_offset, torch.ones_like(xpb) * body_height], dim=-1)

    self.ball_future_pose = torch.where(
        self.mask_before.unsqueeze(-1), self.pos_pred_before, self.pos_pred_after
    )
    # self.mask_invalid = (self.ball_pos[:, 0] < -1.6) | (vx > 0) | (z < 0.7)
    # Invalid mask: use explicit parentheses to avoid bitwise ops on floats
    self.mask_invalid = (
        (self.ball_pos[:, 0] < -1.9)
        | (vx > 0)
        | (z < 0.7)
        | ((self.ball_pos[:, 0] < -1.35) & (vz < 0))
        | self.has_touch_paddle
    )
    self.mask_terminal = (self.ball_pos[:, 0] > -1.5) | (self.ball_pos[:, 0] < -1.9) | self.has_touch_paddle_rew | (vz < 0.0) | (self.ball_pos[:, 2] < 0.6) 
        #mask_terminal: true-> future,mask_terminal: false->distance
    self.has_touch_paddle_rew = self.has_touch_paddle.clone() # finally set mask True for reward computation
    # Expand to match shape (N, 3)
    mask_invalid_expanded = self.mask_invalid.unsqueeze(-1).expand_as(self.ball_future_pose)
    # Zero out those poses
    modified_ball_pos = torch.clone(self.robot_pos)
    modified_ball_pos[:, 1] += paddle_y_offset  # Offset to paddle position
    modified_ball_pos[:, 2] = body_height+ 0.2  # Set z-coordinate to body_height
    self.ball_future_pose = torch.where(
        mask_invalid_expanded,
        modified_ball_pos,
        self.ball_future_pose
    )

    # self.ball_future_pose_vis = self.ball_future_pose + self.scene.env_origins
    mask = self.touched_paddel_no_bounce_table.unsqueeze(-1)  # (N, 1)
    self.ball_future_pose_vis = torch.where(
        mask,
        predict_ball_land_vis,
        self.ball_future_pose,
    ) + self.scene.env_origins
    self.robot_future_pos = torch.where(
        self.mask_before.unsqueeze(-1), self.pos_pred_before_ro, self.pos_pred_after_ro
    )
    self.robot_future_pos = torch.where(
        mask_invalid_expanded,      # [N,3] bool
        self.robot_future_pos.new_tensor([-1.80, 0.3, body_height]).expand_as(self.robot_future_pos),               
        # [-0.9, 0.2, body_height] for all envs
        self.robot_future_pos
    )

    # self.vel_ro_before = torch.clamp((self.pos_pred_before_ro - self.robot_pos ) /torch.clamp(self.t_before.unsqueeze(-1).expand_as(self.ball_future_pose), min=0.2),min=-vel_max, max=vel_max)
    # self.vel_ro_after = torch.clamp((self.pos_pred_after_ro - self.robot_pos ) /torch.clamp(self.t_after.unsqueeze(-1).expand_as(self.ball_future_pose), min=0.2),min=-vel_max, max=vel_max)
    self.vel_ro_before = torch.clamp((self.pos_pred_before_ro - self.robot_pos ) *4,min=-vel_max, max=vel_max)
    self.vel_ro_after = torch.clamp((self.pos_pred_after_ro - self.robot_pos ) *4,min=-vel_max, max=vel_max)

    self.robot_future_vel = torch.where(
        self.mask_before.unsqueeze(-1), self.vel_ro_before, self.vel_ro_after
    )
    self.robot_future_vel = torch.where(
        mask_invalid_expanded,
        torch.zeros_like(self.robot_future_vel),
        self.robot_future_vel
    )

    tb = self.t_before.unsqueeze(-1)  # [N,1]
    ta = self.t_after.unsqueeze(-1)   # [N,1]
    self.ball_future_t = torch.where(self.mask_before.unsqueeze(-1), tb, ta)
    self.ball_future_t = torch.where(
        self.mask_invalid.unsqueeze(-1),
        torch.zeros_like(self.ball_future_t),
        self.ball_future_t
    )

    if not self.headless:
        # self.update_ball_future_visual()
        # self.update_robot_future_pos_visual()
        # self.update_robot_future_vel_visual()
        pass
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
