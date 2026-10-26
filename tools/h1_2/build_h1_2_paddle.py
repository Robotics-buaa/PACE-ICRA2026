"""Create an independent H1-2 paddle layer from the current G1 paddle specs.

Run inside the existing container with IsaacLab/isaaclab.sh -p.
Only initial assembly geometry is checked here; mechanical validation is pending.
"""
from pathlib import Path
import hashlib
import json
import traceback
import uuid

BASE = Path("/workspace/projects/PACE-ICRA2026/legged_lab/assets/unitree")
SOURCE = BASE / "G1_TT/G1_TT.usda"
ROBOT = BASE / "H1_2_BASE/H1_2.usd"
OUTPUT = BASE / "H1_2_TT/H1_2_TT.usda"
REPORT = OUTPUT.parent / "paddle_asset_report.json"
REFERENCE = "../H1_2_BASE/H1_2.usd"
for path in (SOURCE, ROBOT):
    assert path.is_file(), f"INPUT NOT FOUND: {path}"
for path in (OUTPUT, REPORT):
    assert not path.exists(), f"OUTPUT EXISTS; inspect before rebuilding: {path}"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
temporary = OUTPUT.parent / (".build_" + uuid.uuid4().hex + ".usda")

from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateNew(str(temporary))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/h1_2").GetPrim()
    stage.SetDefaultPrim(root)
    root.SetInstanceable(False)
    assert root.GetReferences().AddReference(REFERENCE)
    wrist = stage.GetPrimAtPath("/h1_2/right_wrist_yaw_link")
    assert wrist.IsValid() and wrist.HasAPI(UsdPhysics.RigidBodyAPI), "WRIST BODY MISSING"

    source_layer = Sdf.Layer.FindOrOpen(str(SOURCE))
    assert source_layer is not None, "G1 PADDLE SOURCE OPEN FAILED"
    for source, target in (
        ("/G1/g1_paddle_link", "/h1_2/h1_2_paddle_link"),
        ("/G1/PaddlePhysicsMaterial", "/h1_2/PaddlePhysicsMaterial"),
    ):
        assert source_layer.GetPrimAtPath(source) is not None, f"SPEC MISSING: {source}"
        assert Sdf.CopySpec(source_layer, Sdf.Path(source), stage.GetRootLayer(), Sdf.Path(target))

    paddle_path = Sdf.Path("/h1_2/h1_2_paddle_link")
    material_path = Sdf.Path("/h1_2/PaddlePhysicsMaterial")
    paddle = stage.GetPrimAtPath(paddle_path)
    assert paddle.HasAPI(UsdPhysics.RigidBodyAPI), "PADDLE RIGID BODY API MISSING"
    for name in ("Head", "Handle"):
        shape = stage.GetPrimAtPath(paddle_path.AppendChild(name))
        assert shape.IsValid() and shape.HasAPI(UsdPhysics.CollisionAPI)
        shape.GetRelationship("material:binding:physics").SetTargets([material_path])

    head = UsdGeom.Cylinder(stage.GetPrimAtPath(paddle_path.AppendChild("Head")))
    handle = UsdGeom.Cylinder(stage.GetPrimAtPath(paddle_path.AppendChild("Handle")))
    assert head and handle
    assert head.GetAxisAttr().Get() == "Y" and handle.GetAxisAttr().Get() == "X"
    handle_pos = handle.GetPrim().GetAttribute("xformOp:translate").Get()
    assert abs(handle_pos[1]) < 1e-9 and abs(handle_pos[2]) < 1e-9
    offset = Gf.Vec3d(-handle_pos[0] + handle.GetHeightAttr().Get() / 2.0, 0.0, 0.0)
    cache = UsdGeom.XformCache()
    wrist_world = cache.GetLocalToWorldTransform(wrist)
    root_world = cache.GetLocalToWorldTransform(root)
    paddle_world = Gf.Matrix4d(1.0).SetTranslate(offset) * wrist_world
    transform = paddle.GetAttribute("xformOp:transform")
    assert transform.IsValid(), "PADDLE TRANSFORM OP MISSING"
    transform.Set(paddle_world * root_world.GetInverse())

    joint = UsdPhysics.FixedJoint.Define(stage, "/h1_2/h1_2_paddle_fixed_joint")
    joint.CreateBody0Rel().SetTargets([wrist.GetPath()])
    joint.CreateBody1Rel().SetTargets([paddle_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*offset))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    joint.CreateJointEnabledAttr().Set(True)
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(False)

    cache.Clear()
    actual_world = cache.GetLocalToWorldTransform(paddle)
    error = (wrist_world.Transform(offset) - actual_world.Transform(Gf.Vec3d(0.0))).GetLength()
    rotation_error = max(abs(actual_world[i][j] - wrist_world[i][j]) for i in range(3) for j in range(3))
    assert error < 1e-6 and rotation_error < 1e-6, "INITIAL ASSEMBLY FRAME MISMATCH"
    moving = [p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    assert len(moving) == 27, "EXPECTED 27 REVOLUTE JOINTS"
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert roots == ["/h1_2/pelvis"], "ARTICULATION ROOT CHANGED"
    material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(material_path))
    restitution = material.GetRestitutionAttr().Get()
    mass = UsdPhysics.MassAPI(paddle).GetMassAttr().Get()
    assert abs(restitution - 0.8) < 1e-6 and abs(mass - 0.16) < 1e-6

    report = {
        "base_reference": REFERENCE,
        "g1_paddle_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "h1_2_main_usd_sha256": hashlib.sha256(ROBOT.read_bytes()).hexdigest(),
        "parent_body": str(wrist.GetPath()),
        "paddle_body": str(paddle_path),
        "parent_offset": list(offset),
        "local_normal": [0.0, 1.0, 0.0],
        "head_radius": head.GetRadiusAttr().Get(),
        "head_thickness": head.GetHeightAttr().Get(),
        "mass": mass,
        "restitution": restitution,
        "joint_names": moving,
        "initial_anchor_error_m": error,
        "initial_rotation_matrix_error": rotation_error,
        "validation": "static assembly only; mounting calibration and mechanical test pending",
    }
    commit = ROBOT.parent / "SOURCE_COMMIT.txt"
    if commit.is_file():
        report["unitree_source_commit"] = commit.read_text().strip()
    with OUTPUT.open("x", encoding="utf-8") as stream:
        stream.write(stage.GetRootLayer().ExportToString())
    with REPORT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("Output:", OUTPUT, flush=True)
    print("Base reference:", REFERENCE, flush=True)
    print("Paddle parent:", wrist.GetPath(), flush=True)
    print("Initial parent offset:", tuple(offset), flush=True)
    print("Moving joints:", len(moving), flush=True)
    print("Paddle mass:", mass, "restitution:", restitution, flush=True)
    print("Initial anchor error (m):", error, flush=True)
    print("H1-2 PADDLE ASSEMBLY CHECK PASSED", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    try:
        temporary.unlink(missing_ok=True)
    finally:
        app.close()
