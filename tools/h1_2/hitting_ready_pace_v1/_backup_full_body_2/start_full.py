"""One foreground launch: new wiring check, then original-size PACE budget."""
from datetime import datetime,timezone
import json
from pathlib import Path
import os
import subprocess
import sys
import pace_common as common

common.verify()
common.LOGS.mkdir(parents=True,exist_ok=True)
manifest_sha=common.sha(common.HERE/'manifest.json')
for path in sorted(common.LOGS.glob('full_training_*.json')):
    previous=json.loads(path.read_text())
    if previous.get('manifest_sha256') != manifest_sha:
        continue
    if previous.get('completed') is True:
        print('This package already completed full training; preserved result:',path,flush=True)
        print(previous.get('checkpoint'),flush=True)
        raise SystemExit(0)
    if 'error' not in previous:
        raise SystemExit('An existing full run has no final result; inspect its process/log before starting another: '+str(path))
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
check = common.LOGS/('wiring_check_'+stamp+'.json')
train = common.LOGS/('full_training_'+stamp+'.json')
console = common.LOGS/('full_training_'+stamp+'.log')
base = ['bash','/workspace/projects/IsaacLab/isaaclab.sh','-p',str(common.HERE/'launch.py')]
existing_checks=[]
for path in sorted(common.LOGS.glob('wiring_check_*.json')):
    prior=json.loads(path.read_text())
    if prior.get('manifest_sha256')==manifest_sha and prior.get('completed') is True:
        existing_checks.append(path)
commands=[]
if existing_checks:
    check=existing_checks[-1]
    print('Reuse completed NEW wiring check:',check,flush=True)
else:
    commands.append(base+['--mode','check','--num_envs','8','--iterations','2','--report',str(check)])
commands.append(base+['--mode','train','--num_envs','4096','--iterations','10000',
                     '--check-report',str(check),'--report',str(train)])
print('NEW TASK ONLY: 27 actions, predictor, future body targets. No old training rerun.',flush=True)
print('Step 1: NEW 27-action/predictor wiring check, 8 envs x 2 iterations.',flush=True)
print('Step 2 if successful: 4096 envs x 10000 PPO iterations, foreground.',flush=True)
print('No automatic OOM retries or changes to the training budget.',flush=True)
print('CONSOLE LOG:',console,flush=True)
with console.open('x') as log:
    for command in commands:
        with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,errors='replace',bufsize=1,
                              env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONFAULTHANDLER':'1'}) as process:
            for line in process.stdout:
                log.write(line); log.flush()
                print(line,end='',flush=True)
            code = process.wait()
        report_path = Path(command[command.index('--report')+1])
        stage_result=None
        if report_path.is_file():
            print('RESULT:',report_path,flush=True)
            print(report_path.read_text(),flush=True)
            stage_result=json.loads(report_path.read_text())
        # A native application can exit with status 0 before Python finally.
        # Successful process exit ALONE never authorizes a subsequent stage.
        if code or stage_result is None or stage_result.get('completed') is not True:
            print('PROCESS EXIT CODE:',code,'; completed report required.',flush=True)
            print('STOPPED. No subsequent training stage started. CONSOLE LOG:',console,flush=True)
            raise SystemExit(code or 1)
print('FULL TRAINING REPORT:',train,flush=True)
