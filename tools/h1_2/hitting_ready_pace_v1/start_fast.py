"""Continue an existing full-body run to a total of 10000 iterations."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import pace_common as common

common.verify()
old_sha = 'f7c4f33304c9789c1d415bc389017bb27cc9689814fa6fdfb6d8a75afe789c2e'
checks = []
for path in common.LOGS.glob('wiring_check_*.json'):
    result = json.loads(path.read_text())
    if (result.get('completed') is True and result.get('policy_actions') == 27
            and result.get('manifest_sha256') in (old_sha,common.sha(common.HERE/'manifest.json'))
            and result.get('originals_unchanged') is True
            and result.get('predictor_optimizer_steps',0) > 0
            and result.get('predictor_updates_to_env',0) > 0):
        checks.append(path)
assert checks, 'No completed full-body wiring check found; no training started'
candidates = []
for path in common.LOGS.glob('*/pace_ready_progress.json'):
    if not any(tag in path.parent.name for tag in ('ready_pace_train_4096x10000_','ready_pace_train_fast_')):
        continue
    result = json.loads(path.read_text())
    checkpoint = Path(result['checkpoint'])
    assert checkpoint.resolve().parent == path.parent.resolve(), 'Checkpoint escaped run directory'
    if result.get('policy_actions') == 27 and result.get('predictor_optimizer_steps',0) > 0 and checkpoint.is_file():
        candidates.append((int(result['iteration']),path.stat().st_mtime,checkpoint))
assert candidates, 'No saved full-body training checkpoint found; no restart from scratch'
iteration,_,checkpoint = max(candidates)
if iteration >= 9999:
    raise SystemExit('The saved full-body run already reached iteration 9999; preserved: '+str(checkpoint))
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
report = common.LOGS/('fast_training_'+stamp+'.json')
console = common.LOGS/('fast_training_'+stamp+'.log')
command = ['bash','/workspace/projects/IsaacLab/isaaclab.sh','-p',str(common.HERE/'launch.py'),
    '--mode','train','--num_envs','4096','--iterations','10000',
    '--resume-checkpoint',str(checkpoint),'--check-report',str(sorted(checks)[-1]),'--report',str(report)]
print('RESUME:',checkpoint,flush=True)
print('Completed iterations:',iteration+1,'; remaining:',10000-(iteration+1),flush=True)
print('Reuse completed full-body check. First GPU block parity checked before any new PPO update.',flush=True)
print('Policy, predictor and both optimizers restored; environment histories restart at ready pose.',flush=True)
print('CONSOLE:',console,flush=True)
with console.open('x') as log:
    with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
            errors='replace',bufsize=1,env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONFAULTHANDLER':'1'}) as process:
        for line in process.stdout:
            print(line,end='',flush=True)
            log.write(line);log.flush()
        code = process.wait()
result = json.loads(report.read_text()) if report.is_file() else {}
print('REPORT:',report,flush=True)
if code or result.get('completed') is not True:
    raise SystemExit(code or 1)
