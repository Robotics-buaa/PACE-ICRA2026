"""Add bilateral tasks, back up registration, preserve all verified baseline files."""
import ast
import base64
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import zlib

ROOT = Path("/workspace/projects/PACE-ICRA2026")
data = json.loads(zlib.decompress(base64.b64decode("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")))
registry = ROOT / "legged_lab/envs/__init__.py"
original = registry.read_bytes()
text = original.decode("utf-8")
registration = data["registration"]
registered = "# BEGIN PACE H1-2 BILATERAL TASKS" in text
if registered:
    assert registration in text, "EXISTING BILATERAL REGISTRATION DIFFERS"
else:
    assert "h1_2_tt_bilateral" not in text, "BILATERAL TASK NAME ALREADY EXISTS"
normalized = lambda s: s.replace("\r\n", "\n").rstrip() + "\n"
for name, expected in data["source_sha256"].items():
    actual = (ROOT / name).read_text(encoding="utf-8")
    if name == "legged_lab/envs/__init__.py" and registered:
        actual = actual.replace(registration, "")
    assert hashlib.sha256(normalized(actual).encode()).hexdigest() == expected, "BASELINE CHANGED; INSPECT BEFORE DEPLOYMENT: " + name
previous = json.loads((ROOT / "tools/h1_2/task_install_report.json").read_text())
check = json.loads((ROOT / "legged_lab/assets/unitree/H1_2_TT/task_check_report.json").read_text())
assert check["task_check_passed"], "PREVIOUS H1-2 TASK CHECK MUST PASS"
protected = dict(previous["protected_sha256"])
for name, digest in protected.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, "ORIGINAL PROTECTED FILE CHANGED: " + name
for name in data["source_sha256"]:
    if name != "legged_lab/envs/__init__.py":
        protected[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
for name, content in data["files"].items():
    ast.parse(content, filename=name)
    target = ROOT / name
    target.relative_to(ROOT)
    assert not target.exists() or target.read_text(encoding="utf-8") == content, "NEW FILE CONFLICT; PRESERVED: " + name
new_text = text if registered else text.rstrip() + "\n\n" + registration
ast.parse(new_text)
report_path = ROOT / "tools/h1_2/bilateral_install_report.json"
backup = None
if registered and report_path.is_file():
    backup = json.loads(report_path.read_text())["registration_backup"]
if not registered:
    backup_path = ROOT / ("tools/h1_2/backups/envs_init_before_bilateral_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".py")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(registry, backup_path)
    backup = str(backup_path)
for name, content in data["files"].items():
    target = ROOT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with target.open("x", encoding="utf-8", newline="\n") as f:
            f.write(content)
if not registered:
    assert registry.read_bytes() == original, "REGISTRY CHANGED DURING DEPLOYMENT"
    fd, temporary = tempfile.mkstemp(prefix=".h1_2_bilateral_", dir=registry.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(new_text.encode())
        os.chmod(temporary, stat.S_IMODE(registry.stat().st_mode))
        os.replace(temporary, registry)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
for name, digest in protected.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, "PROTECTED FILE CHANGED: " + name
report = {
    "protected_sha256": protected,
    "new_files_sha256": {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in data["files"]},
    "registration_backup": backup,
    "tasks": ["h1_2_tt_bilateral", "h1_2_tt_bilateral_eval"],
    "experiment_name": "h1_2_table_tennis_bilateral",
    "remote_interface_check": "pending", "training": "not started",
    "scope": "approximate serve-region filtering and central stance guidance; no style/face reward or skill claim",
}
report_path.write_text(json.dumps(report, indent=2) + "\n")
print("Registration backup:", backup)
print("Baseline files preserved:", len(protected))
print("Tasks: h1_2_tt_bilateral / h1_2_tt_bilateral_eval")
print("Experiment: h1_2_table_tennis_bilateral; new initialization; training not started")
print("H1-2 BILATERAL FILES READY")
