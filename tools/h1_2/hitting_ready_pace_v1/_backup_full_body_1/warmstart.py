"""Transfer the standing actor as a trainable prior, never a joint override."""
import numpy as np


def standing_affine(residual, paddle_goal, base_goal, action_scale, ball_scale,
                    robot_scale, perception_scale):
    a, b = np.eye(99), np.zeros(99)
    b[60:87] = np.asarray(residual)*action_scale
    a[87:98] = 0.
    b[93:96] = np.asarray(paddle_goal)*ball_scale
    b[96:98] = np.asarray(base_goal[:2])*robot_scale
    a[96,90] = -robot_scale/perception_scale
    a[97,91] = -robot_scale/perception_scale
    return a,b


def initialize_actor(runner, env, checkpoint, config):
    import torch
    saved = torch.load(str(checkpoint),map_location='cpu',weights_only=False)
    original_cfg = torch.load(str(config),map_location='cpu',weights_only=False)
    assert original_cfg['empirical_normalization'] is False and runner.empirical_normalization is False, 'Normalized actor transfer requires separate mapping'
    state = saved['model_state_dict']
    actor = {k.removeprefix('actor.'):v.clone() for k,v in state.items() if k.startswith('actor.')}
    assert actor and actor['0.weight'].shape[1] == 495
    origin = (env.table.data.root_link_pos_w-env.scene.env_origins)[0].cpu().numpy()
    base = np.asarray([*env.cfg.ready_base_xy,float(env.cfg.reference_base_height)])
    paddle = base+env._h1_2_reference_paddle_offset_b[0].cpu().numpy()
    residual = ((env.action_center[0]-env.robot.data.default_joint_pos[0,env.action_joint_ids])/env.action_scale).cpu().numpy()
    a,b = standing_affine(residual,paddle,base-origin,env.obs_scales.actions,
                         env.obs_scales.ball_pos,env.obs_scales.robot_pos,env.obs_scales.perception)
    old = actor['0.weight'].clone()
    for frame in range(5):
        sl = slice(frame*99,(frame+1)*99)
        actor['0.weight'][:,sl] = old[:,sl]@old.new_tensor(a)
        actor['0.bias'] += old[:,sl]@old.new_tensor(b)
    last = max(int(k.split('.')[0]) for k in actor if k.endswith('.weight'))
    right = [env.action_joint_names.index(n) for n in __import__('ready_pose').RIGHT_READY_RAD]
    actor[str(last)+'.weight'][right] = 0.
    actor[str(last)+'.bias'][right] = 0.
    runner.alg.policy.actor.load_state_dict(actor,strict=True)
    # Preserve the learned exploration scale as initial prior; all parameters
    # including every leg/torso weight remain trainable under original PPO.
    with torch.no_grad():
        for key in ('std','log_std'):
            if key in state and hasattr(runner.alg.policy,key):
                getattr(runner.alg.policy,key).copy_(state[key].to(runner.device))
    assert all(p.requires_grad for p in runner.alg.policy.parameters())
    runner.warmstart_audit = dict(source=str(checkpoint), actor_only=True,
        critic='fresh', predictor='fresh', right_head='zero at ready',
        joint_override=False, all_27_actions_trainable=True,
        normalizers='identity; original checkpoint untouched',
        note='initial actor prior, not a proof of balance while moving')
    print('[PACE-READY] STANDING ACTOR PRIOR:',runner.warmstart_audit,flush=True)
