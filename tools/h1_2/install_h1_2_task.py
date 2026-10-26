"""Deploy only new H1-2 task files and append backed-up task registration."""
from pathlib import Path
import ast
import base64
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zlib

ROOT = Path("/workspace/projects/PACE-ICRA2026")
data = json.loads(zlib.decompress(base64.b64decode("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")))
files = data["files"]
registration = data["registration"]
registry = ROOT / "legged_lab/envs/__init__.py"
original_registry = registry.read_bytes()
text = original_registry.decode("utf-8")
normalized = lambda s: s.replace("\r\n", "\n").rstrip() + "\n"
has_registration = "# BEGIN PACE H1-2 TASKS" in text
if has_registration:
    assert registration in text, "EXISTING H1-2 REGISTRATION DIFFERS"
else:
    assert "h1_2_tt" not in text, "H1-2 NAME ALREADY IN REGISTRY; INSPECT IT"
for name, expected in data["source_sha256"].items():
    actual = (ROOT / name).read_text(encoding="utf-8")
    if name == "legged_lab/envs/__init__.py" and has_registration:
        actual = actual.replace(registration, "")
    digest = hashlib.sha256(normalized(actual).encode()).hexdigest()
    assert digest == expected, "SOURCE CHANGED SINCE HANDOFF: " + name
for name in (
    "legged_lab/assets/unitree/h1_2_tt.py",
    "legged_lab/assets/unitree/H1_2_BASE/joint_spec.json",
    "legged_lab/assets/unitree/H1_2_TT/H1_2_TT.usda",
    "legged_lab/assets/unitree/H1_2_TT/wrist_mechanical_report.json",
):
    assert (ROOT / name).is_file(), "PREVIOUS H1-2 STEP MISSING: " + name
mechanics = json.loads((ROOT / "legged_lab/assets/unitree/H1_2_TT/wrist_mechanical_report.json").read_text())
assert len(mechanics["tests"]) == 12 and mechanics["max_anchor_error_m"] < 0.002
for name, source in files.items():
    ast.parse(source, filename=name)
    target = ROOT / name
    target.relative_to(ROOT)
    assert not target.exists() or target.read_text(encoding="utf-8") == source, "NEW FILE ALREADY EXISTS WITH DIFFERENT CONTENT: " + name
new_text = text if has_registration else text.rstrip() + "\n\n" + registration
ast.parse(new_text, filename=str(registry))

protected_names = [name for name in data["source_sha256"] if name != "legged_lab/envs/__init__.py"] + [
    "legged_lab/physics/aerodynamics.py", "legged_lab/mdp/rewards.py",
    "legged_lab/assets/unitree/g1_tt.py", "legged_lab/assets/unitree/g1_tt_wrist.py",
    "legged_lab/assets/unitree/G1_TT/G1_TT.usda", "legged_lab/assets/unitree/G1_TT_WRIST/G1_TT_WRIST.usda",
    "legged_lab/envs/g1_tt_wrist/g1_tt_wrist_config.py", "legged_lab/envs/g1_tt_wrist/g1_tt_wrist_env.py",
    "legged_lab/assets/unitree/h1_2_tt.py", "legged_lab/assets/unitree/H1_2_BASE/H1_2.usd",
    "legged_lab/assets/unitree/H1_2_TT/H1_2_TT.usda",
]
protected = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in protected_names}
backup = None
if has_registration:
    previous_report = ROOT / "tools/h1_2/task_install_report.json"
    if previous_report.is_file():
        previous_backup = json.loads(previous_report.read_text())["registration_backup"]
        backup = Path(previous_backup) if previous_backup else None
if not has_registration:
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    backup = ROOT / ("tools/h1_2/backups/envs_init_" + stamp + ".py")
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(registry, backup)
for name, source in files.items():
    target = ROOT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(source)
    print("Ready:", target)
if not has_registration:
    assert registry.read_bytes() == original_registry, "REGISTRY CHANGED DURING DEPLOYMENT"
    fd, temporary = tempfile.mkstemp(prefix=".h1_2_register_", dir=registry.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(new_text.encode("utf-8"))
        os.chmod(temporary, stat.S_IMODE(registry.stat().st_mode))
        os.replace(temporary, registry)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
for name, digest in protected.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, "PROTECTED FILE CHANGED: " + name
manifest = {
    "protected_sha256": protected,
    "new_files_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
    "source_sha256_normalized": data["source_sha256"],
    "registration_backup": str(backup) if backup else None,
    "tasks": ["h1_2_tt", "h1_2_tt_eval"],
    "experiment_name": "h1_2_table_tennis",
    "head_distance_reward_disabled": True,
    "task_simulation_check": "pending",
}
(ROOT / "tools/h1_2/task_install_report.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Registration backup:", backup if backup else "already registered")
print("Original source/assets preserved:", len(protected))
print("H1-2 TASK FILES READY")
