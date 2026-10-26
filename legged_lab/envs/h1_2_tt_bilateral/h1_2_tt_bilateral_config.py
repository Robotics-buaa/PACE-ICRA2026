"""Initial, moderate bilateral serves; baseline H1-2 configs remain intact."""
from copy import deepcopy
from isaaclab.utils import configclass
from legged_lab.envs.h1_2_tt.h1_2_tt_config import (
    H1_2TableTennisEnvCfg, H1_2TT_EvalEnvCfg, H1_2TableTennisAgentCfg,
)


def _bilateral_settings(cfg):
    # Copy mutable inherited configs before modifying them.
    cfg.ball = deepcopy(cfg.ball)
    cfg.domain_rand = deepcopy(cfg.domain_rand)
    cfg.scene = deepcopy(cfg.scene)
    cfg.scene.ball.init_state.pos = (1.35, 0.0, 1.03)
    cfg.ball.ball_speed_x_range = (-6.5, -5.0)
    cfg.ball.ball_speed_y_range = (-0.65, 0.65)  # envelope, not uniform across zero
    cfg.ball.ball_speed_z_range = (1.5, 2.0)
    cfg.ball.ball_pos_y_range = (-0.05, 0.05)
    cfg.ball.ball_reset_repeat = 1
    cfg.domain_rand.events.reset_base.params["pose_range"] = {
        "x": (-0.40, -0.30), "y": (-0.05, 0.05), "yaw": (-0.10, 0.10),
    }


@configclass
class H1_2BilateralEnvCfg(H1_2TableTennisEnvCfg):
    ready_base_xy: tuple = (-1.80, 0.0)
    bilateral_speed_y_abs: tuple = (0.35, 0.65)
    bilateral_hit_y_abs: tuple = (0.22, 0.40)
    bilateral_hit_x: tuple = (-1.85, -1.65)
    bilateral_table_margin: float = 0.03
    bilateral_base_y_gain: float = 0.25
    bilateral_base_y_limit: float = 0.10
    bilateral_max_sample_rounds: int = 128

    def __post_init__(self):
        super().__post_init__()
        _bilateral_settings(self)


@configclass
class H1_2BilateralEvalEnvCfg(H1_2TT_EvalEnvCfg):
    ready_base_xy: tuple = (-1.80, 0.0)
    bilateral_speed_y_abs: tuple = (0.35, 0.65)
    bilateral_hit_y_abs: tuple = (0.22, 0.40)
    bilateral_hit_x: tuple = (-1.85, -1.65)
    bilateral_table_margin: float = 0.03
    bilateral_base_y_gain: float = 0.25
    bilateral_base_y_limit: float = 0.10
    bilateral_max_sample_rounds: int = 128

    def __post_init__(self):
        super().__post_init__()
        _bilateral_settings(self)


@configclass
class H1_2BilateralAgentCfg(H1_2TableTennisAgentCfg):
    experiment_name: str = "h1_2_table_tennis_bilateral"
    resume: bool = False
