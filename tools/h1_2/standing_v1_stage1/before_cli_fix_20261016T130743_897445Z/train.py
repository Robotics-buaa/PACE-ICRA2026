"""Explicit, fixed-budget standing training; automatically evaluate only after success."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import time
import traceback

from common import (EXPERIMENT, HERE, check_runtime, sha, snapshot_originals,
                    verify_originals, write_json)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description='Separate standing stage: 512 envs, 500 PPO iterations, seed42.')
parser.add_argument('--execute-training', action='store_true', help='Actually start this fixed-budget training stage.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.execute_training:
    parser.error('No training started. Use --execute-training only to start the reviewed 512x500 standing stage.')
assert not args.distributed, 'THIS STAGE IS SINGLE GPU ONLY'
originals = snapshot_originals()
stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_%fZ')
run_dir = EXPERIMENT/(stamp+'_stand_v1_512x500_seed42')
run_dir.mkdir(parents=True, exist_ok=False)
report = {'status': 'preparing', 'task': 'h1_2_stand_v1', 'num_envs': 512,
          'iterations': 500, 'num_steps_per_env': 24, 'seed': 42, 'resume': False,
          'loaded_checkpoint': None, 'predictor': False, 'original_hashes_before': originals,
          'log_dir': str(run_dir), 'evaluation': 'pending; training is not a skill acceptance'}
write_json(run_dir/'stage1_training_report.json', report)
app = env = runner = None
success = False
started = time.time()
try:
    app = AppLauncher(args).app
    import numpy as np
    import random
    import torch
    from isaaclab.utils.io import dump_yaml
    import legged_lab.envs
    from legged_lab.utils import task_registry
    from rsl_rl.runners import OnPolicyRunner
    from standing_v1.task import register_tasks

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    register_tasks()
    cfg, agent = task_registry.get_cfgs('h1_2_stand_v1')
    cfg, agent = deepcopy(cfg), deepcopy(agent)
    cfg.scene.num_envs, cfg.scene.seed, cfg.device = 512, 42, args.device
    agent.device, agent.seed, agent.resume = args.device, 42, False
    agent.experiment_name, agent.run_name = 'h1_2_standing_v1', 'stand_v1_512x500_seed42'
    agent.max_iterations, agent.num_steps_per_env, agent.save_interval = 500, 24, 100
    agent.logger = 'tensorboard'
    train_cfg = agent.to_dict()
    assert train_cfg['algorithm']['class_name'] == 'PPO'
    assert train_cfg['policy']['class_name'] == 'ActorCritic', 'THIS STAGE EXPECTS FEED-FORWARD PPO'
    assert train_cfg['resume'] is False and train_cfg['num_steps_per_env'] == 24
    Parent = task_registry.get_task_class('h1_2_stand_v1')

    class FiniteTrainEnv(Parent):
        def step(self, actions):
            assert torch.isfinite(actions).all(), 'NONFINITE TRAINING ACTION'
            obs, reward, done, extras = super().step(actions)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all(), 'NONFINITE TRAINING TRANSITION'
            assert torch.isfinite(extras['observations']['critic']).all(), 'NONFINITE CRITIC OBSERVATION'
            return obs, reward, done, extras

    env = FiniteTrainEnv(cfg, args.headless)
    report['runtime_lower_kp'] = check_runtime(env)
    report['reference_height_m'] = float(cfg.reference_base_height)
    report['step_dt'] = env.step_dt
    dump_yaml(str(run_dir/'params/env.yaml'), cfg)
    dump_yaml(str(run_dir/'params/agent.yaml'), deepcopy(train_cfg))
    torch.save(deepcopy(train_cfg), str(run_dir/'runner_config.pt'))
    report['runner_config_sha256'] = sha(run_dir/'runner_config.pt')
    # The observed runner mutates nested config dicts with pop(), so retain the
    # complete saved config above and pass an independent copy to construction.
    runner = OnPolicyRunner(env, deepcopy(train_cfg), log_dir=str(run_dir), device=args.device)
    assert runner.current_learning_iteration == 0 and runner.training_type == 'rl'
    before = {k: v.detach().cpu().clone() for k, v in runner.alg.policy.state_dict().items()}
    print('[STAND-TRAIN] Fixed budget: 512 environments x 500 iterations x 24 steps; seed42', flush=True)
    print('[STAND-TRAIN] New random policy; no checkpoint load; base PPO; no predictor', flush=True)
    print('[STAND-TRAIN] RUN_DIR:', run_dir, flush=True)
    report['status'] = 'training'
    write_json(run_dir/'stage1_training_report.json', report)
    runner.learn(num_learning_iterations=500, init_at_random_ep_len=False)
    assert runner.current_learning_iteration == 499, 'UNEXPECTED FINAL ITERATION'
    after = {k: v.detach().cpu() for k, v in runner.alg.policy.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in after.values()), 'NONFINITE TRAINED MODEL'
    assert any(not torch.equal(before[k], v) for k, v in after.items()), 'NO POLICY UPDATE OBSERVED'
    optimizer_steps = [float(state['step']) for state in runner.alg.optimizer.state.values() if 'step' in state]
    assert optimizer_steps and max(optimizer_steps) > 0, 'NO OPTIMIZER STEP OBSERVED'
    checkpoint = run_dir/'model_499.pt'
    assert checkpoint.is_file(), 'FINAL CHECKPOINT NOT SAVED'
    verify_originals(originals)
    report.update(status='training_completed', checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                  final_iteration=499, optimizer_max_step=max(optimizer_steps), policy_changed=True,
                  control_transitions=512*24*500, wall_seconds=time.time()-started,
                  original_files_unchanged=True)
    write_json(run_dir/'stage1_training_report.json', report)
    print('[STAND-TRAIN] TRAINING COMPLETED:', checkpoint, flush=True)
    print('[STAND-TRAIN] Next: fresh-process 8-env, first-episode 20s inference evaluation.', flush=True)
    success = True
except BaseException as error:
    report.update(status='failed', error=repr(error), traceback=traceback.format_exc(), wall_seconds=time.time()-started)
    try:
        verify_originals(originals)
        report['original_files_unchanged'] = True
    except BaseException as protection_error:
        report['protection_error'] = repr(protection_error)
    write_json(run_dir/'stage1_training_report.json', report)
    traceback.print_exc()
    raise
finally:
    try:
        if runner is not None and runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if app is not None:
                app.close()

if success:
    # Replace this process so the training GPU context is released before eval.
    os.environ['OMNI_KIT_ALLOW_ROOT'] = '1'
    launcher = '/workspace/projects/IsaacLab/isaaclab.sh'
    os.execv(launcher, [launcher, '-p', str(HERE/'evaluate.py'), '--headless',
                       '--device', args.device, '--run-dir', str(run_dir)])
