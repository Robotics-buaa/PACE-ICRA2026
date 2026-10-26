"""Run the unmodified PACE train module with task-local registration/runner."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import importlib
import faulthandler
import sys
import traceback
import pace_common as common

parser = argparse.ArgumentParser()
parser.add_argument('--mode',choices=('check','train'),required=True)
parser.add_argument('--num_envs',type=int,required=True)
parser.add_argument('--iterations',type=int,required=True)
parser.add_argument('--check-report',type=Path)
parser.add_argument('--report',type=Path,required=True)
parser.add_argument('--resume-checkpoint',type=Path)
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
    assert accepted['manifest_sha256'] in (manifest_sha,'f7c4f33304c9789c1d415bc389017bb27cc9689814fa6fdfb6d8a75afe789c2e')
    assert accepted['predictor_optimizer_steps'] > 0 and accepted['predictor_updates_to_env'] > 0
    assert accepted['policy_actions'] == 27 and accepted['originals_unchanged'] is True
if args.resume_checkpoint is not None:
    assert args.mode == 'train' and args.iterations == 10000
    checkpoint = args.resume_checkpoint.resolve(strict=True)
    checkpoint.relative_to(common.LOGS.resolve())
    progress = json.loads((checkpoint.parent/'pace_ready_progress.json').read_text())
    assert progress['policy_actions'] == 27 and progress['predictor_optimizer_steps'] > 0
    assert Path(progress['checkpoint']).resolve() == checkpoint
    if 'checkpoint_sha256' in progress:
        assert common.sha(checkpoint) == progress['checkpoint_sha256']
    common.RESUME_CHECKPOINT,common.RESUME_PROGRESS = checkpoint,progress
common.LOGS.mkdir(parents=True,exist_ok=True)
assert not args.report.exists(), 'Refuse to replace an existing launch report'
report = dict(completed=False,mode=args.mode,num_envs=args.num_envs,iterations=args.iterations,
              manifest_sha256=manifest_sha,policy_actions=27,originals_unchanged=None,
              originals_unchanged_at_start=True,
              original_training_entry='legged_lab.scripts.train',
              original_runner='OnPolicyPredictorRegressionRunner with task-local boundary/observation corrections')
common.write_report(args.report,report)
common.ACTIVE_REPORT,common.ACTIVE_STATE = args.report,report
original_module = None
faulthandler.enable(all_threads=True)
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
        '--run_name=ready_pace_'+('train_fast' if args.resume_checkpoint else args.mode)+'_'+str(args.num_envs)+'x'+str(args.iterations)+'_'+datetime.now(timezone.utc).strftime('%H%M%S_%f'),
        '--logger=tensorboard']
    print('[PACE-READY] ORIGINAL ENTRY:', ' '.join(sys.argv),flush=True)
    # Keep the original module/AppLauncher globals referenced through failure
    # reporting. Running it as __main__ let its application shutdown happen
    # inside temporary runpy globals before our final result was persisted.
    common.phase('IMPORT_ORIGINAL_TRAIN_MODULE')
    original_module = importlib.import_module('legged_lab.scripts.train')
    original_dump = original_module.dump_yaml
    def dump_with_phase(*values,**options):
        common.phase('DUMP_YAML_BEGIN',path=str(values[0]))
        result = original_dump(*values,**options)
        common.phase('DUMP_YAML_END',path=str(values[0]))
        return result
    original_module.dump_yaml = dump_with_phase
    common.phase('CALL_ORIGINAL_TRAIN')
    original_module.train()
    common.phase('ORIGINAL_TRAIN_RETURNED')
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
        resumed_from=str(common.RESUME_CHECKPOINT) if common.RESUME_CHECKPOINT else None,
        checkpoint_sha256=common.sha(final),totals=env.total,
        control_transitions=args.num_envs*runner.num_steps_per_env*runner.session_iterations,
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
    print('[PACE-READY] ERROR:',repr(error),flush=True)
    traceback.print_exc()
    raise
finally:
    # Cleanup and reporting must occur BEFORE SimulationApp.close(), which may
    # terminate the process directly. Parent also validates this report.
    if common.ENV is not None and not getattr(common.ENV,'_is_closed',True):
        try:
            common.ENV.close()
        except BaseException as cleanup_error:
            report['cleanup_error'] = repr(cleanup_error)
    common.write_report(args.report,report)
    print('[PACE-READY] LAUNCH REPORT:',args.report,flush=True)
    if original_module is not None:
        try:
            original_module.simulation_app.close()
        except BaseException as close_error:
            if not (isinstance(close_error,SystemExit) and close_error.code in (0,None)):
                report.update(completed=False,shutdown_error=repr(close_error))
                common.write_report(args.report,report)
                if 'error' not in report:
                    raise
