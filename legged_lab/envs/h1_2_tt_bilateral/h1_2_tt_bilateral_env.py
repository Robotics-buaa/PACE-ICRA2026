"""Bilateral serve sampler and central stance guidance for one right paddle.

The sampler uses PACE's existing approximate prediction equations. Its accepted
regions are predicted regions, not measured PhysX contacts or stroke labels.
"""
import torch
from legged_lab.envs.h1_2_tt.h1_2_tt_env import H1_2TTEnv


def approximate_serve_regions(position, velocity, k):
    """Match TTEnv's BEFORE-bounce forecast, without its X clamp or invalid mask."""
    g, h = 9.81, 0.78
    vz = velocity[:, 2]
    discriminant = vz.square() + 2 * g * (position[:, 2] - h)
    t_land = (vz + discriminant.clamp_min(0).sqrt()) / g
    vz_up = (0.9 * (g * t_land - vz)).clamp_min(0)
    k = max(float(k), 1e-8)
    t_up = torch.atan(vz_up * (k / g) ** 0.5) / (g * k) ** 0.5
    def displacement(v, t):
        return torch.sign(v) * torch.log1p(k * v.abs() * t[:, None]) / k
    landing_xy = position[:, :2] + displacement(velocity[:, :2], t_land)
    apex_xy = landing_xy + displacement(0.7 * velocity[:, :2], t_up)
    apex_z = h + torch.log1p(k / g * vz_up.square()) / (2 * k)
    return landing_xy, torch.cat((apex_xy, apex_z[:, None]), dim=-1)


class H1_2BilateralEnv(H1_2TTEnv):
    def init_buffers(self):
        super().init_buffers()
        if self.cfg.ball.ball_reset_repeat != 1:
            raise ValueError("Bilateral task requires fresh serves: ball_reset_repeat=1")
        # Persistent across robot resets. Each env sees equal counts over each pair.
        self._next_serve_side = torch.where(
            torch.rand(self.num_envs, device=self.device) < 0.5, -1.0, 1.0,
        )
        self.serve_side = torch.zeros(self.num_envs, device=self.device)
        self.serve_landing_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.serve_apex = torch.zeros((self.num_envs, 3), device=self.device)
        self.serve_count_by_side = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)

    def _sample_bilateral_states(self, env_ids, sides):
        state = self.ball.data.default_root_state[env_ids].clone()
        pending = torch.arange(len(env_ids), device=self.device)
        # TTEnv.compute_perception defines ball_pos relative to env_origins;
        # its contact bounds and ball forecast use that same environment frame.
        # A table rigid-body origin can differ from the table surface center.
        # Only the base target below converts to the table-root frame.
        landing = state.new_zeros((len(env_ids), 2))
        apex = state.new_zeros((len(env_ids), 3))
        margin = self.cfg.bilateral_table_margin
        xmin, xmax = self.cfg.table.table_own_contact_x
        ymin, ymax = self.cfg.table.table_own_contact_y
        def uniform(n, limits):
            return torch.empty(n, device=self.device).uniform_(*limits)
        for _ in range(self.cfg.bilateral_max_sample_rounds):
            n = len(pending)
            if n == 0:
                break
            candidate = self.ball.data.default_root_state[env_ids[pending]].clone()
            candidate[:, 1] += uniform(n, self.cfg.ball.ball_pos_y_range)
            candidate[:, 7] = uniform(n, self.cfg.ball.ball_speed_x_range)
            candidate[:, 8] = sides[pending] * uniform(n, self.cfg.bilateral_speed_y_abs)
            candidate[:, 9] = uniform(n, self.cfg.ball.ball_speed_z_range)
            candidate[:, 10:] = 0.0
            predicted_land, predicted_apex = approximate_serve_regions(
                candidate[:, :3], candidate[:, 7:10], self.ball_drag_k,
            )
            signed_y = predicted_apex[:, 1] * sides[pending]
            accept = (
                (predicted_land[:, 0] >= xmin + margin)
                & (predicted_land[:, 0] <= xmax - margin)
                & (predicted_land[:, 1] >= ymin + margin)
                & (predicted_land[:, 1] <= ymax - margin)
                & (predicted_apex[:, 0] >= self.cfg.bilateral_hit_x[0])
                & (predicted_apex[:, 0] <= self.cfg.bilateral_hit_x[1])
                & (signed_y >= self.cfg.bilateral_hit_y_abs[0])
                & (signed_y <= self.cfg.bilateral_hit_y_abs[1])
            )
            selected = pending[accept]
            state[selected] = candidate[accept]
            landing[selected] = predicted_land[accept]
            apex[selected] = predicted_apex[accept]
            pending = pending[~accept]
        if len(pending):
            raise RuntimeError("Bilateral sampling exhausted; inspect bounds instead of serving invalid balls")
        state[:, :3] += self.scene.env_origins[env_ids]
        return state, landing, apex

    def reset_ball(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        sides = self._next_serve_side[env_ids].clone()
        # Validate/sample first; base reset retains all original flag bookkeeping.
        state, landing, apex = self._sample_bilateral_states(env_ids, sides)
        super().reset_ball(env_ids)
        self.reset_ball_state_buf[env_ids] = state
        self.ball.write_root_pose_to_sim(state[:, :7], env_ids)
        self.ball.write_root_velocity_to_sim(state[:, 7:], env_ids)
        self.serve_side[env_ids] = sides
        self.serve_landing_xy[env_ids] = landing
        self.serve_apex[env_ids] = apex
        self.serve_count_by_side[env_ids, (sides > 0).long()] += 1
        self._next_serve_side[env_ids] *= -1

    def _h1_2_base_target(self, ball_target):
        # Use only the supplied forecast and current heading, never sampler labels.
        # Actor passes its learned forecast; privileged critic/rewards keep theirs.
        table_origin = self.table.data.root_link_pos_w - self.scene.env_origins
        target = ball_target.clone() - table_origin
        target[:, 0] -= self._h1_2_paddle_offset_xy()[:, 0]
        center_y = self.cfg.ready_base_xy[1] - table_origin[:, 1]
        delta_y = (target[:, 1] - center_y) * self.cfg.bilateral_base_y_gain
        target[:, 1] = center_y + delta_y.clamp(
            -self.cfg.bilateral_base_y_limit, self.cfg.bilateral_base_y_limit,
        )
        target[:, 2] = self.cfg.reference_base_height - table_origin[:, 2]
        return target
