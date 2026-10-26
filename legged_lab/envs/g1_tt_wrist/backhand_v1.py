"""Calibrated preparation-only backhand experiment for PACE G1 wrist.

Install as legged_lab/envs/g1_tt_wrist/backhand_v1.py. At the END of
legged_lab/envs/__init__.py append: from .g1_tt_wrist import backhand_v1

Registers g1_tt_wrist_backhand_v1 and g1_tt_wrist_backhand_v1_eval.
Uses the user's 2026-10-22 prepare and impact calibration. Keeps action and
observation dimensions and action-centering defaults unchanged. Impact XYZ
is used ONLY for the base's horizontal stand-off target; impact joint angles
and normal are recorded below but are not imposed as fixed contact targets.

Smoke-test 32 envs / 20 updates, then short-train 1024 envs / 1000 updates.
Keep --predictor and load the existing wrist predictor checkpoint. The
experiment name stays g1_table_tennis_wrist to find the existing checkpoint.
Use a distinct --run_name to create a new run directory.

Source basis: Robotics-buaa/PACE-ICRA2026 feat/pjh-run. This file is syntax
checked only; PyTorch and Isaac Sim execution require the user's environment.
"""

from copy import deepcopy

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .g1_tt_wrist_env import G1WristTTEnv
from .g1_tt_wrist_config import (
    G1WristTableTennisEnvCfg,
    G1WristTT_EvalEnvCfg,
    G1WristTableTennisAgentCfg,
)
from legged_lab.utils.task_registry import task_registry


PREPARE_POS_B = (0.193292588, 0.214408949, 0.169944778)
IMPACT_POS_HEADING = (0.306128144, 0.200727522, 0.108592272)
PREPARE_NORMAL_B = (-0.760521531, -0.166645840, 0.627564192)
IMPACT_NORMAL_B = (-0.954479337, 0.000606060, -0.298277885)

# Explicit ordering: never index joint state by hard-coded articulation IDs.
PREPARE_JOINTS = {
    "right_shoulder_pitch_joint": -0.69,
    "right_shoulder_roll_joint": -0.25,
    "right_shoulder_yaw_joint": 1.14,
    "right_elbow_pitch_joint": 0.10,
    "right_elbow_roll_joint": 1.46,
    "right_palm_joint": 0.44,
    "torso_joint": 0.06,
}
IMPACT_JOINTS = {
    "right_shoulder_pitch_joint": -1.03,
    "right_shoulder_roll_joint": -0.26,
    "right_shoulder_yaw_joint": 1.07,
    "right_elbow_pitch_joint": 0.47,
    "right_elbow_roll_joint": 0.62,
    "right_palm_joint": 0.52,
    "torso_joint": 0.06,
}


def calibrated_prepare_reward(env, asset_cfg, target_pos_b, target_joints,
                              position_std, joint_std):
    """Soft position + joint reward, released before the contact phase.

    ball_future_t is the BASE ENV's analytical future-target time proxy;
    it is not a guaranteed actual time-to-contact or the learned predictor's
    time output. Diagnostics below report whether its gate is ever active.
    """
    robot = env.scene[asset_cfg.name]
    base_q = robot.data.root_link_quat_w
    base_q = base_q / base_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    pos_b = math_utils.quat_rotate_inverse(
        base_q, env.paddle_touch_point - robot.data.root_link_pos_w,
    )
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    if q.shape[-1] != len(target_joints):
        raise RuntimeError("Backhand joint matching/order mismatch")
    pos_ref = pos_b.new_tensor(target_pos_b)
    q_ref = q.new_tensor(target_joints)
    position_error = ((pos_b - pos_ref) / position_std).square().sum(-1)
    joint_error = ((q - q_ref) / joint_std).square().mean(-1)
    score = torch.exp(-0.5 * (position_error + joint_error))

    tau = env.ball_future_t.squeeze(-1)
    gate = ((~env.mask_invalid) & (~env.has_touch_paddle)
            & (tau > 0.25) & (tau < 0.70))
    return torch.where(gate, torch.nan_to_num(score, nan=0.0),
                       torch.zeros_like(score))


def configure_backhand(cfg):
    # Initial serving condition, NOT a forehand/backhand label. Change this
    # single value to the feasible serving vy established by your evaluation.
    serve_vy = 0.2
    cfg.ball.ball_speed_x_range = (-5.8, -5.8)
    cfg.ball.ball_speed_y_range = (serve_vy, serve_vy)
    cfg.ball.ball_speed_z_range = (1.7, 1.7)
    cfg.ball.ball_pos_y_range = (0.0, 0.0)

    # Existing G1 implementation uses only this offset's XY coordinates.
    # Setting its Z value does NOT change the predicted contact height.
    cfg.reference_paddle_offset_b = IMPACT_POS_HEADING

    # Match initial base randomization in train/eval for this first experiment.
    ref = G1WristTT_EvalEnvCfg()
    cfg.domain_rand.events.reset_base = deepcopy(ref.domain_rand.events.reset_base)

    cfg.reward = deepcopy(cfg.reward)
    # Replace any earlier prototype term using the same name.
    cfg.reward.backhand_prepare = RewTerm(
        func=calibrated_prepare_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(PREPARE_JOINTS), preserve_order=True,
            ),
            "target_pos_b": PREPARE_POS_B,
            "target_joints": tuple(PREPARE_JOINTS.values()),
            "position_std": 0.20,
            "joint_std": 0.50,
        },
    )


@configclass
class BackhandV1TrainCfg(G1WristTableTennisEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_backhand(self)


@configclass
class BackhandV1EvalCfg(G1WristTT_EvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_backhand(self)


class BackhandV1Env(G1WristTTEnv):
    """Adds console diagnostics only; policy interface is unchanged."""

    def step(self, actions):
        result = super().step(actions)
        count = getattr(self, "_bh_debug_steps", 0) + 1
        self._bh_debug_steps = count
        tau = self.ball_future_t.squeeze(-1)
        valid = (~self.mask_invalid) & (~self.has_touch_paddle) & torch.isfinite(tau)
        active = valid & (tau > 0.25) & (tau < 0.70)
        self._bh_debug_total = getattr(self, "_bh_debug_total", 0) + tau.numel()
        self._bh_debug_active = getattr(self, "_bh_debug_active", 0) + int(active.sum())
        if count % 100 == 0:
            fraction = self._bh_debug_active / max(self._bh_debug_total, 1)
            msg = f"[Backhand] prep_gate_fraction={fraction:.4f}"
            if bool(valid.any()):
                heights = self.ball_future_pose[valid, 2]
                heights = heights[torch.isfinite(heights)]
                if heights.numel():
                    msg += (f" | analytic_target_z median={heights.median().item():.3f}"
                            f" range=[{heights.min().item():.3f},"
                            f" {heights.max().item():.3f}]")
            print(msg, flush=True)
            self._bh_debug_total = 0
            self._bh_debug_active = 0
        return result


task_registry.register(
    "g1_tt_wrist_backhand_v1", BackhandV1Env,
    BackhandV1TrainCfg(), G1WristTableTennisAgentCfg(),
)
task_registry.register(
    "g1_tt_wrist_backhand_v1_eval", BackhandV1Env,
    BackhandV1EvalCfg(), G1WristTableTennisAgentCfg(),
)
