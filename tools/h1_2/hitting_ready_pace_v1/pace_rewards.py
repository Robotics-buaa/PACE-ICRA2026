"""Only task-local physical event/ready-reference/safety additions."""
import torch


def ready_arm_deviation(env, asset_cfg):
    ids = asset_cfg.joint_ids
    return (env.robot.data.joint_pos[:, ids]-env.ready_q[:, ids]).abs().sum(-1)


def early_margin(env):
    return ((env.safety.soft-env.safety.minimum).clamp_min(0.)/env.safety.soft).square().sum(-1)


def invalid_contact(env):
    return (env.events.pulses['invalid'] | env.events.pulses['double']).float()


def actual_return(env):
    return env.events.pulses['return'].float()


def body_height(env):
    h = env.robot.data.root_link_pos_w[:, 2]-env.scene.env_origins[:, 2]
    return torch.exp(-((h-float(env.cfg.reference_base_height))/.05).square())
