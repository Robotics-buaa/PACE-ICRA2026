"""Read calibrated JSON and add phased joints + signed paddle-normal rewards.

Install next to backhand_v1.py; append at END of legged_lab/envs/__init__.py:
    from .g1_tt_wrist import backhand_v2

Default calibration: <repo>/debug_poses/g1_wrist_poses_v2.json.
Keys are explicitly selected below (never silently choose a numbered snapshot).
If only impact is in the new file, the original v1 prepare is retained.
Missing JSON skips v2 registration without breaking v1 tasks. Invalid existing
JSON raises an explicit error. The reference snapshot is embedded in env.yaml.

No changes to policy dimensions, action centers, ball physics or predictor
architecture. Reuses v1 serving setup. Reward weights/window widths below are
initial tuning values, not validated optimums. ball_future_t is an analytical
future-target-time proxy, not a measured time-to-contact. Interpolated joint
references are not a certified collision-free or dynamically feasible motion.

Syntax and pure JSON/phase helpers checked here. Torch/Isaac runtime must be
smoke-tested in the user's existing PACE environment before longer training.
"""

from copy import deepcopy
import json
import math
from pathlib import Path

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from legged_lab.utils.task_registry import task_registry
from . import backhand_v1 as v1


REPO_ROOT = Path(__file__).resolve().parents[3]
POSE_FILE = REPO_ROOT / "debug_poses/g1_wrist_poses_v2.json"
PREPARE_KEY = "prepare"
IMPACT_KEY = "impact"  # Set to "impact_2" if that is your intended snapshot.

SWING_START = 0.30  # Begin transitioning at this analytical time proxy [s].
SWING_END = 0.10    # Reference becomes the impact pose by this time [s].
STYLE_WINDOW = 0.70
JOINT_STD = 0.50    # Soft joint matching tolerance [rad].
NORMAL_DISTANCE_STD = 0.25  # Normal reward near the ball [m].


def finite_vector(value, size, label):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} must contain {size} numbers")
    out = tuple(float(x) for x in value)
    if not all(math.isfinite(x) for x in out):
        raise ValueError(f"{label} contains non-finite values")
    return out


def read_pose(item, label):
    if item.get("length_unit") != "m" or item.get("angle_unit") != "rad":
        raise ValueError(f"{label}: expected length_unit=m, angle_unit=rad")
    pos = finite_vector(item["paddle_position_b"], 3, f"{label}.position")
    normal = finite_vector(item["paddle_normal_b"], 3, f"{label}.normal")
    magnitude = math.sqrt(sum(x*x for x in normal))
    if magnitude < 1e-6:
        raise ValueError(f"{label}: zero paddle normal")
    normal = tuple(x / magnitude for x in normal)
    joints = item.get("adjustable_joints_rad", item.get("all_joints_rad", {}))
    q = finite_vector([joints[name] for name in v1.PREPARE_JOINTS],
                      len(v1.PREPARE_JOINTS), f"{label}.joints")
    heading = finite_vector(item.get("paddle_position_heading", pos), 3,
                            f"{label}.position_heading")
    return {"position_b": pos, "normal_b": normal,
            "joints": q, "position_heading": heading}


def load_reference(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if IMPACT_KEY not in data:
        raise ValueError(f"Missing key {IMPACT_KEY!r} in {path}; keys={list(data)}")
    impact = read_pose(data[IMPACT_KEY], IMPACT_KEY)
    if PREPARE_KEY in data:
        prepare = read_pose(data[PREPARE_KEY], PREPARE_KEY)
    else:
        prepare = {
            "position_b": tuple(v1.PREPARE_POS_B),
            "normal_b": tuple(v1.PREPARE_NORMAL_B),
            "joints": tuple(v1.PREPARE_JOINTS.values()),
            "position_heading": tuple(v1.PREPARE_POS_B),
        }
    return {"prepare": prepare, "impact": impact,
            "source_path": str(path), "impact_key": IMPACT_KEY,
            "prepare_key": PREPARE_KEY if PREPARE_KEY in data else "v1_fallback"}


def smooth_phase(tau):
    # Works with torch.Tensor; scalar reference is also tested independently.
    u = ((SWING_START - tau) / (SWING_START - SWING_END)).clamp(0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def phase_and_gate(env):
    tau = env.ball_future_t.squeeze(-1)
    finite = torch.isfinite(tau)
    safe_tau = torch.nan_to_num(tau, nan=STYLE_WINDOW + 1.0)
    alpha = smooth_phase(safe_tau)
    valid = (finite & (~env.mask_invalid) & (~env.has_touch_paddle)
             & (safe_tau >= 0.0) & (safe_tau < STYLE_WINDOW))
    return alpha, valid


def base_quaternion(env):
    quat = env.robot.data.root_link_quat_w
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def phased_joint_reward(env, asset_cfg):
    ref = env.cfg.backhand_v2_reference
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]

    # 始终参考准备姿态，不再插值到击球姿态。
    q_prepare = q.new_tensor(ref["prepare"]["joints"])

    if q.shape[-1] != q_prepare.shape[-1]:
        raise RuntimeError("Prepare joint reference dimension mismatch")

    alpha, valid = phase_and_gate(env)

    error = ((q - q_prepare) / JOINT_STD).square().mean(-1)
    score = torch.exp(-0.5 * error)

    # 接近击球时逐渐减弱准备约束，避免强迫保持引拍姿态。
    prepare_weight = 1.0 - alpha

    return torch.where(
        valid,
        prepare_weight * torch.nan_to_num(score, nan=0.0),
        torch.zeros_like(score),
    )


def prepare_position_reward(env):
    ref = env.cfg.backhand_v2_reference
    pos = math_utils.quat_rotate_inverse(
        base_quaternion(env),
        env.paddle_touch_point - env.robot.data.root_link_pos_w,
    )
    target = pos.new_tensor(ref["prepare"]["position_b"])
    alpha, valid = phase_and_gate(env)
    score = torch.exp(-0.5 * ((pos - target) / 0.20).square().sum(-1))
    return torch.where(valid, (1.0 - alpha) * torch.nan_to_num(score, nan=0.0),
                       torch.zeros_like(score))


def signed_normal_cosine(env):
    normal = math_utils.quat_rotate_inverse(base_quaternion(env), env.paddle_normal)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    target = normal.new_tensor(env.cfg.backhand_v2_reference["impact"]["normal_b"])
    return (normal * target).sum(-1).clamp(-1.0, 1.0)


def impact_normal_reward(env):
    # Keep sign: opposite paddle faces must not receive identical rewards.
    cosine = signed_normal_cosine(env)
    distance = (env.ball.data.root_pos_w - env.paddle_touch_point).norm(dim=-1)
    alpha, valid = phase_and_gate(env)
    proximity = torch.exp(-0.5 * (distance / NORMAL_DISTANCE_STD).square())
    score = alpha * proximity * (1.0 + cosine) * 0.5
    return torch.where(valid, torch.nan_to_num(score, nan=0.0),
                       torch.zeros_like(score))

def excessive_base_yaw(env, tolerance=0.52):
    # 当前球桌布局中，机器人朝 +世界X 为面向对方。
    yaw = env.robot.data.heading_w
    error = torch.atan2(torch.sin(yaw), torch.cos(yaw))

    # 允许约 ±30° 的自然侧身，仅惩罚超出部分。
    excess = torch.clamp(torch.abs(error) - tolerance, min=0.0)
    return excess.square()

def horizontal_handle_reward(env, asset_cfg):
    robot = env.scene[asset_cfg.name]

    # body_ids 顺序：手掌、球拍。
    palm_pos = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    paddle_pos = robot.data.body_pos_w[:, asset_cfg.body_ids[1]]

    direction = paddle_pos - palm_pos
    direction = direction / direction.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)

    # 世界坐标系中，长轴越水平，z分量越接近0。
    score = 1.0 - direction[:, 2].square()

    alpha, valid = phase_and_gate(env)
    return torch.where(
        valid,
        alpha * score,
        torch.zeros_like(score),
    )

def paddle_center_velocity_w(env):
    """球拍中心的世界坐标系速度，包含转动产生的速度。"""
    state = env.robot.data.body_com_state_w[
        :, env.paddle_body_id
    ]

    com_pos = state[:, :3]
    com_vel = state[:, 7:10]
    angular_vel = state[:, 10:13]

    offset = env.paddle_touch_point - com_pos

    return com_vel + torch.cross(
        angular_vel, offset, dim=-1
    )


def incoming_ball_gate(env):
    """仅在来球接近球拍、尚未判定击球时生效。"""
    distance = (
        env.ball.data.root_pos_w - env.paddle_touch_point
    ).norm(dim=-1)

    incoming = env.ball.data.root_lin_vel_w[:, 0] < -0.1

    valid = (
        incoming
        & (~env.has_touch_paddle)
        & (distance < 0.4)
    )

    proximity = torch.exp(
        -0.5 * (distance / 0.25).square()
    )

    return valid, proximity

def forward_swing_reward(env):
    velocity = paddle_center_velocity_w(env)

    vx = velocity[:, 0]
    vy = velocity[:, 1]
    vz = velocity[:, 2]

    # 鼓励向前，达到一定速度后不再额外奖励加速。
    forward = torch.clamp(vx / 1.5, min=0.0, max=1.0)

    # 减少纯侧向甩拍，允许竖直方向有一定调整空间。
    lateral = torch.exp(-0.5 * (vy / 1.0).square())
    vertical = torch.exp(
        -0.5 * ((vz - 0.2) / 1.0).square()
    )

    valid, proximity = incoming_ball_gate(env)
    score = proximity * forward * lateral * vertical

    return torch.where(
        valid,
        torch.nan_to_num(score, nan=0.0),
        torch.zeros_like(score),
    )

def downward_swing_penalty(env):
    vz = paddle_center_velocity_w(env)[:, 2]

    # 允许轻微下降，只惩罚明显向下的挥拍。
    downward = torch.clamp(
        (-vz - 0.3) / 1.0,
        min=0.0,
        max=3.0,
    )

    valid, proximity = incoming_ball_gate(env)
    penalty = proximity * downward.square()

    return torch.where(
        valid,
        torch.nan_to_num(penalty, nan=0.0),
        torch.zeros_like(penalty),
    ) 

def paddle_open_angle(env):
    """返回拍面开角（弧度）及朝向对方的水平对齐程度。"""
    normal = env.paddle_normal
    normal = normal / normal.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)

    # 同一拍面有两个相反法向，统一选择朝 +X 的一个。
    normal = torch.where(
        normal[:, 0:1] < 0.0,
        -normal,
        normal,
    )

    horizontal = normal[:, :2].norm(
        dim=-1
    ).clamp_min(1e-8)

    # 正角度：朝向对方的法向带有向上分量。
    angle = torch.atan2(normal[:, 2], horizontal)

    # 避免拍面朝侧面，却仅靠倾角满足要求。
    alignment = torch.clamp(
        normal[:, 0] / horizontal,
        min=0.0,
        max=1.0,
    )

    return angle, alignment


def paddle_open_angle_reward(
    env,
    target_deg=10.0,
    std_deg=10.0,
):
    angle, alignment = paddle_open_angle(env)

    target = math.radians(target_deg)
    std = math.radians(std_deg)

    angle_score = torch.exp(
        -0.5 * ((angle - target) / std).square()
    )

    # 复用已有窗口：仅在来球接近、尚未击球时生效。
    valid, proximity = incoming_ball_gate(env)

    score = proximity * alignment * angle_score

    return torch.where(
        valid,
        torch.nan_to_num(score, nan=0.0),
        torch.zeros_like(score),
    )

def configure_v2(cfg, reference):
    v1.configure_backhand(cfg)
    cfg.backhand_v2_reference = deepcopy(reference)
    cfg.reference_paddle_offset_b = tuple(reference["impact"]["position_heading"])
    # Avoid imposing the old preparation reference during the swing.
    cfg.reward.backhand_prepare.weight = 0.0
    cfg.reward.backhand_phase_joints = RewTerm(
        func=phased_joint_reward, weight=1.0,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=list(v1.PREPARE_JOINTS), preserve_order=True,
        )},
    )
    # cfg.reward.backhand_prepare_position = RewTerm(
    #     func=prepare_position_reward, weight=0.5,
    # )
    cfg.reward.backhand_impact_normal = RewTerm(
        func=impact_normal_reward, weight=1.0,
    )
    # 关闭旧的固定标定法向奖励，避免目标冲突。
    cfg.reward.backhand_impact_normal.weight = 0.0

    # 鼓励接触前略微向上开拍。
    cfg.reward.backhand_open_angle = RewTerm(
        func=paddle_open_angle_reward,
        weight=1.0,
        params={
            "target_deg": 10.0,
            "std_deg": 10.0,
        },
    )

    cfg.reward.backhand_base_yaw = RewTerm(
        func=excessive_base_yaw,
        weight=-1.0,
        params={"tolerance": 0.52},
    )
    cfg.reward.backhand_horizontal_handle = RewTerm(
        func=horizontal_handle_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "right_palm_link",
                    "g1_paddle_link",
                ],
                preserve_order=True,
            ),
        },
    )
    # 默认基座位置：只将 y 改为0，保留原来的 x 和高度。
    x0, _, z0 = cfg.scene.robot.init_state.pos
    cfg.scene.robot.init_state.pos = (x0, 0.0, z0)

    # 重置位置随机偏移的 y。
    reset_params = cfg.domain_rand.events.reset_base.params
    reset_params["pose_range"]["y"] = (0.0, 0.0)

    # 初始侧向随机速度。
    reset_params["velocity_range"]["y"] = (0.0, 0.0)


    # 保留准备站位的 x，将 y 设置为中心线0。
    cfg.ready_base_xy = (cfg.ready_base_xy[0], 0.0)

    # 提高过网奖励权重：默认100 → 1000。
    cfg.reward.reward_future_pass_net.weight = 1000.0

    cfg.reward.backhand_forward_swing = RewTerm(
        func=forward_swing_reward,
        weight=2.0,
    )

    cfg.reward.backhand_downward_swing = RewTerm(
        func=downward_swing_penalty,
        weight=-1.0,
    )

    # 加强向前、略向上送拍的引导。
    cfg.reward.backhand_forward_swing.weight = 5.0

    # 加强拍面略微朝上的引导。
    cfg.reward.backhand_open_angle.weight = 5.0


@configclass
class BackhandV2TrainCfg(v1.G1WristTableTennisEnvCfg):
    backhand_v2_reference: dict = None

    def __post_init__(self):
        super().__post_init__()
        configure_v2(self, load_reference(POSE_FILE))


@configclass
class BackhandV2EvalCfg(v1.G1WristTT_EvalEnvCfg):
    backhand_v2_reference: dict = None

    def __post_init__(self):
        super().__post_init__()
        configure_v2(self, load_reference(POSE_FILE))


class BackhandV2Env(v1.G1WristTTEnv):
    def step(self, actions):
        result = super().step(actions)
        alpha, valid = phase_and_gate(self)
        distance = (self.ball.data.root_pos_w - self.paddle_touch_point).norm(dim=-1)
        near = valid & (distance < NORMAL_DISTANCE_STD) & (alpha > 0.5)
        count = getattr(self, "_bh2_steps", 0) + 1
        self._bh2_steps = count
        self._bh2_count = getattr(self, "_bh2_count", 0) + int(near.sum())
        self._bh2_total = getattr(self, "_bh2_total", 0) + self.num_envs
        cosine = signed_normal_cosine(self)
        self._bh2_cos_sum = getattr(self, "_bh2_cos_sum", 0.0) + float(cosine[near].sum())
        if count % 100 == 0:
            n = self._bh2_count
            value = f"{self._bh2_cos_sum/n:.3f}" if n else "NA"
            print(f"[BH2] near_gate_fraction={n/max(self._bh2_total, 1):.4f}"
                  f" | near_normal_cos={value}", flush=True)
            self._bh2_count = 0
            self._bh2_total = 0
            self._bh2_cos_sum = 0.0

        # A first positive distance-based hit score is a CONTACT CANDIDATE,
        # not verified physical collision. Useful for pose-height diagnostics.
        if self.num_envs == 1:
            close = bool(self.ball_contact_rew[0] > 0.0)
            previous = getattr(self, "_bh2_previous_close", False)
            done = bool(result[2][0])
            if close and not previous and not done:
                print(f"[BH2 candidate] paddle_z={self.paddle_touch_point[0, 2].item():.3f}"
                      f" base_z={self.robot.data.root_link_pos_w[0, 2].item():.3f}"
                      f" normal_cos={cosine[0].item():.3f}", flush=True)
                angle, _ = paddle_open_angle(self)
                print(
                    "[BH2 candidate] open_angle_deg="
                    f"{torch.rad2deg(angle[0]).item():.1f}",
                    flush=True,
                )
            self._bh2_previous_close = close and not done
        return result


if POSE_FILE.is_file():
    reference = load_reference(POSE_FILE)
    print(f"[BH2] Calibration: {POSE_FILE}; impact={IMPACT_KEY}; "
          f"prepare={reference['prepare_key']}", flush=True)
    task_registry.register("g1_tt_wrist_backhand_v2", BackhandV2Env,
                           BackhandV2TrainCfg(), v1.G1WristTableTennisAgentCfg())
    task_registry.register("g1_tt_wrist_backhand_v2_eval", BackhandV2Env,
                           BackhandV2EvalCfg(), v1.G1WristTableTennisAgentCfg())
else:
    print(f"[BH2] Not registered: save calibration JSON to {POSE_FILE}", flush=True)
