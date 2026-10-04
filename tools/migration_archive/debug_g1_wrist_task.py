"""Expose errors and progress in the existing wrist task check, without editing it."""
import ast
import sys
from pathlib import Path


def message(text):
    return ast.parse("print(%r, flush=True)" % text).body[0]


def instrument(source, filename):
    tree = ast.parse(source, filename=filename)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    fn = functions["check_task"]
    outer = next(n for n in fn.body if isinstance(n, ast.Try))
    if outer.handlers:
        raise RuntimeError("Unexpected existing check_task exception handlers; source preserved")
    # The original finally closes the simulator. Print the error before cleanup
    # even if simulator shutdown prevents Python's usual uncaught-error output.
    handler = ast.parse('''
try:
    pass
except BaseException:
    import traceback
    print("[FAIL] Wrist task check raised an exception before cleanup:", flush=True)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    raise
''').body[0].handlers[0]
    outer.handlers.append(handler)
    body = []
    for stmt in outer.body:
        if isinstance(stmt, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "runner" for t in stmt.targets
        ):
            body.append(message("[CHECK] Constructing policy and predictor runner"))
        if isinstance(stmt, ast.With):
            body.append(message("[CHECK] Policy/predictor forward pass and 24 control steps"))
            for child in stmt.body:
                if isinstance(child, ast.For):
                    child.body.insert(0, ast.parse(
                        'if _ in (0, 23):\n'
                        '    print("[CHECK] Control step %d/24" % (_ + 1), flush=True)'
                    ).body[0])
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value.func
            if isinstance(call, ast.Attribute) and call.attr == "reset":
                body.append(message("[CHECK] Resetting wrist state and target in environment 0"))
        body.append(stmt)
    outer.body = body
    return compile(ast.fix_missing_locations(tree), filename, "exec")


def main():
    script = Path.cwd() / "convert_g1_palm_joint.py"
    if not script.is_file():
        raise FileNotFoundError("Run from PACE project root containing convert_g1_palm_joint.py")
    code = instrument(script.read_text(encoding="utf-8"), str(script))
    namespace = {"__name__": "g1_wrist_debug", "__file__": str(script)}
    exec(code, namespace)
    sys.argv = [str(script), "--stage", "task"]
    print("[INFO] Running existing task check with exception reporting; no source edits", flush=True)
    namespace["main"]()


if __name__ == "__main__":
    main()

