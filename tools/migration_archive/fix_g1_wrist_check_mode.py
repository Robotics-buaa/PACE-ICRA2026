"""Replace the wrist check's inference context; leave all task/asset files intact."""
import ast
import shutil
from datetime import datetime
from pathlib import Path

p = Path("convert_g1_palm_joint.py")
source = p.read_text(encoding="utf-8")
tree = ast.parse(source)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check_task")
lines = source.splitlines(keepends=True)
section = "".join(lines[fn.lineno - 1:fn.end_lineno])
old = "with torch.inference_mode():"
new = "with torch.no_grad():"
if section.count(old) == 1:
    fixed = section.replace(old, new, 1)
    updated = "".join(lines[:fn.lineno - 1]) + fixed + "".join(lines[fn.end_lineno:])
    compile(updated, str(p), "exec")
    backup = p.with_name(p.name + ".no_grad_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".bak")
    shutil.copy2(p, backup)
    p.write_text(updated, encoding="utf-8")
    print("Backup:", backup)
    print("[PASS] check_task now uses torch.no_grad()")
elif old not in section and section.count(new) == 1:
    print("[PASS] check_task already uses torch.no_grad(); no changes")
else:
    raise RuntimeError("Unexpected check_task content; no changes made")
