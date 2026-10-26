"""Read-only baseline verification and new-run outputs only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/workspace/projects/PACE-ICRA2026')
HERE = Path(__file__).resolve().parent
RUN = ROOT/'logs/h1_2_standing_v1/2026-10-16_13-07-50_805816Z_stand_v1_512x500_seed42'
EXPERIMENT = ROOT/'logs/h1_2_hitting_ready_v1'
STAND_SHA = 'e44584088aeff823d6e8d2c593f298f436236e7b2d3d75086f00610dd174d035'
for p in (ROOT, HERE.parent, HERE.parent/'standing_v1_stage1', HERE.parent/'hitting_front_v1'):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(HERE))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    package = json.loads((HERE/'manifest.json').read_text())
    for name, digest in package['files_sha256'].items():
        assert sha(HERE/name) == digest, 'NEW PACKAGE CHANGED: '+name
    for name, digest in package['required_files_sha256'].items():
        assert sha(ROOT/name) == digest, 'PROTECTED DEPENDENCY CHANGED: '+name
    # Existing protection manifests remain authoritative, never rewritten.
    from common import verify as verify_standing
    verify_standing()
    assert sha(RUN/'model_499.pt') == STAND_SHA
    training = json.loads((RUN/'stage1_training_report.json').read_text())
    assert training['status'] == 'training_completed'
    assert sha(RUN/'runner_config.pt') == training['runner_config_sha256']
    return package


def write_json(path, obj):
    path = Path(path)
    path.resolve().relative_to(EXPERIMENT.resolve())
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def new_output(prefix):
    folder = EXPERIMENT/(prefix+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    folder.mkdir(parents=True, exist_ok=False)
    return folder
