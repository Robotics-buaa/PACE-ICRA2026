"""Repair the new sampler's table-root restriction; preserve original tasks/assets."""
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
report_path = ROOT / "tools/h1_2/bilateral_install_report.json"
report_bytes = report_path.read_bytes()
report = json.loads(report_bytes)
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for name, sha in report["protected_sha256"].items():
    assert digest(ROOT / name) == sha, "BASELINE FILE CHANGED: " + name
for name, sha in report["new_files_sha256"].items():
    if name not in data["files"]:
        assert digest(ROOT / name) == sha, "OTHER BILATERAL FILE CHANGED: " + name
original = {}
changed = []
for name, source in data["files"].items():
    ast.parse(source, filename=name)
    target = ROOT / name
    target.relative_to(ROOT)
    original[name] = target.read_bytes()
    sha = digest(target)
    assert sha in (data["old_sha256"][name], data["new_sha256"][name]), "REMOTE FILE DIFFERS; PRESERVED: " + name
    assert report["new_files_sha256"].get(name) in (data["old_sha256"][name], data["new_sha256"][name]), "MANIFEST VERSION DIFFERS: " + name
    if sha != data["new_sha256"][name]:
        changed.append(name)
backup_dir = None
if changed:
    backup_dir = ROOT / ("tools/h1_2/backups/bilateral_coordinate_fix_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(report_path, backup_dir / "bilateral_install_report.json")
    for name in changed:
        backup = backup_dir / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, backup)
def atomic_write(path, content):
    fd, temporary = tempfile.mkstemp(prefix=".bilateral_coordinate_fix_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
for name in changed:
    target = ROOT / name
    assert target.read_bytes() == original[name], "FILE CHANGED DURING PATCH: " + name
    atomic_write(target, data["files"][name].encode())
    assert digest(target) == data["new_sha256"][name]
for name, sha in report["protected_sha256"].items():
    assert digest(ROOT / name) == sha, "PROTECTED FILE CHANGED: " + name
report["new_files_sha256"].update(data["new_sha256"])
report["coordinate_fix_applied"] = True
report["remote_interface_check"] = "pending after coordinate fix"
if backup_dir:
    report.setdefault("coordinate_fix_backups", []).append(str(backup_dir))
assert report_path.read_bytes() == report_bytes, "MANIFEST CHANGED DURING PATCH"
atomic_write(report_path, (json.dumps(report, indent=2) + "\n").encode())
print("Coordinate fix: ball forecasts/contact bounds stay in environment frame; base target stays in table-root frame")
print("Backup:", backup_dir if backup_dir else "fix already present")
print("Baseline source/assets unchanged; no learning")
print("H1-2 BILATERAL COORDINATE FIX READY")
