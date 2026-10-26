"""Run the unmodified PACE train module with task-local registration/runner."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import runpy
import sys
import traceback
import pace_common as common

parser = argparse.ArgumentParser()
parser.add_argument('--mode',choices=('check','train'),required=True)
parser.add_argument('--num_envs',type=int,required=True)
parser.add_argument('--iterations',type=int,required=True)
parser.add_argument('--check-report',type=Path)
parser.add_argument('--report',type=Path,required=True)
args = parser.parse_args()
assert 1 <= args.num_envs <= 4096 and args.iterations > 0
if args.mode == 'check':
    assert args.num_envs == 8 and args.iterations == 2, 'New wiring check is exactly 8x2'
common.CHECK_MODE = args.mode == 'check'
manifest = common.verify()
manifest_sha = common.sha(common.HERE/'manifest.json')
if args.mode == 'train':
    import json
    accepted = json.loads(args.check_report.read_text())
    assert accepted['mode'] == 'check' and accepted['completed'] is True
    assert accepted['manifest_sha256'] == manifest_sha
    assert accepted['predictor_optimizer_steps'] > 0 and accepted['predictor_updates_to_env'] > 0
    assert accepted['policy_actions'] == 27 and accepted['originals_unchanged'] is True
common.LOGS.mkdir(parents=True,exist_ok=True)
assert not args.report.exists(), 'Refuse to replace an existing launch report'
report = dict(completed=False,mode=args.mode,num_envs=args.num_envs,iterations=args.iterations,
              manifest_sha256=manifest_sha,policy_actions=27,originals_unchanged=False,
              original_training_entry='legged_lab.scripts.train',
              original_runner='OnPolicyPredictorRegressionRunner with task-local boundary/observation corrections')
common.write_report(args.report,report)
os.chdir(common.ROOT)

try:
    # These imports also occur before AppLauncher in original train.py. The
    # environment/config imports remain lazy until its application has started.
    import rsl_rl.runners as runners
    from pace_runner import PaceReadyRunner
    from legged_lab.utils import task_registry
    original_get_cfgs = task_registry.get_cfgs
    def configs(name):
        if name == common.TASK and name not in task_registry.task_classes:
            from pace_env import PaceReadyEnv,make_configs
            cfg,agent = make_configs()
            task_registry.register(name,PaceReadyEnv,cfg,agent)
        return original_get_cfgs(name)
    task_registry.get_cfgs = configs
    runners.OnPolicyPredictorRegressionRunner = PaceReadyRunner
    sys.argv = ['legged_lab.scripts.train','--task='+common.TASK,
        '--num_envs='+str(args.num_envs),'--max_iterations='+str(args.iterations),
        '--predictor','--headless','--device=cuda:0','--seed=42',
        '--experiment_name=h1_2_table_tennis_ready_pace',
        '--run_name=ready_pace_'+args.mode+'_'+str(args.num_envs)+'x'+str(args.iterations)+'_'+datetime.now(timezone.utc).strftime('%H%M%S_%f'),
        '--logger=tensorboard']
    print('[PACE-READY] ORIGINAL ENTRY:', ' '.join(sys.argv),flush=True)
    runpy.run_module('legged_lab.scripts.train',run_name='__main__')
    import torch
    runner,env = common.RUNNER,common.ENV
    assert runner is not None and env is not None and common.CLOSED_VERIFIED
    assert runner.learn_completed
    assert any(not torch.equal(v.cpu(),runner.initial_actor[k]) for k,v in runner.alg.policy.actor.state_dict().items())
    assert runner.predictor_optimizer_steps > 0 and runner.predictor_updates_to_env > 0
    assert any(not torch.equal(v.cpu(),runner.initial_predictor[k]) for k,v in runner._predictor.state_dict().items())
    final = Path(runner.log_dir)/'model_final.pt'
    saved = torch.load(str(final),map_location='cpu',weights_only=False)
    assert 'pred_state_dict' in saved and 'pred_optimizer_state_dict' in saved
    assert all(torch.isfinite(v).all() for v in saved['model_state_dict'].values())
    common.verify()
    report.update(completed=True,originals_unchanged=True,checkpoint=str(final),
        checkpoint_sha256=common.sha(final),totals=env.total,
        control_transitions=args.num_envs*runner.num_steps_per_env*args.iterations,
        predictor_optimizer_steps=runner.predictor_optimizer_steps,
        predictor_updates_to_env=runner.predictor_updates_to_env,warmstart=runner.warmstart_audit,
        verdict='NEW_FULL_BODY_PACE_WIRING_PASS' if args.mode == 'check' else 'TRAINING_COMPLETED; evaluate physical hits/returns separately')
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc())
    if common.ENV is not None and common.ENV.last_evidence:
        evidence=common.ENV.last_evidence
        keys=('episode_step','guard_failure','no_path','min_margins','body_peak','height','tilt','robot_position_table')
        report['last_transition_summary'] = {k:evidence[k].detach().cpu().tolist() for k in keys}
        failed=evidence['guard_failure'].nonzero(as_tuple=False).flatten()[:2]
        report['failed_env_details']=[dict(env=int(i),**{k:v[i].detach().cpu().tolist() for k,v in evidence.items()}) for i in failed]
    raise
finally:
    common.write_report(args.report,report)
    print('[PACE-READY] LAUNCH REPORT:',args.report,flush=True)
