"""Independent PACE integration; verify all received upstream source hashes."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/workspace/projects/PACE-ICRA2026')
HERE = Path(__file__).resolve().parent
LOGS = ROOT/'logs/h1_2_table_tennis_ready_pace'
STAND = ROOT/'logs/h1_2_standing_v1/2026-10-16_13-07-50_805816Z_stand_v1_512x500_seed42'
STAND_SHA = 'e44584088aeff823d6e8d2c593f298f436236e7b2d3d75086f00610dd174d035'
TASK = 'h1_2_tt_ready_pace'
ENV = RUNNER = None
BINDINGS = {}
CLOSED_VERIFIED = False
CHECK_MODE = False
for path in (ROOT, HERE):
    sys.path.insert(0, str(path))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    manifest = json.loads((HERE/'manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        assert sha(HERE/name) == digest, 'NEW PACKAGE CHANGED: '+name
    for path, digest in manifest['protected_files_sha256'].items():
        assert sha(path) == digest, 'UPSTREAM CHANGED: '+path
    assert sha(STAND/'model_499.pt') == STAND_SHA
    # Preserve authoritative original protection checks, including old assets.
    sys.path.insert(0, str(ROOT/'tools/h1_2/hitting_ready_v1'))
    from paths import verify as verify_ready
    verify_ready()
    sys.path.insert(0, str(HERE))
    return manifest


def write_report(path, data):
    path = Path(path)
    path.resolve().relative_to(LOGS.resolve())
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)
