"""Build the G1 paddle asset and check its attachment and ball rebound."""

import ast
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path('/workspace/projects/PACE-ICRA2026')
sys.path.insert(0, str(PROJECT))

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

try:
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from isaaclab_assets import G1_CFG

    asset_dir = PROJECT / 'legged_lab/assets/unitree'
    usd_dir = asset_dir / 'G1_TT'
    usd_dir.mkdir(parents=True, exist_ok=True)
    output = usd_dir / 'G1_TT.usda'
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if output.exists():
        shutil.copy2(output, output.with_name(output.name + '.' + stamp + '.bak'))

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/G1').GetPrim()
    assert root.GetReferences().AddReference(G1_CFG.spawn.usd_path)
    root.SetInstanceable(False)
    stage.SetDefaultPrim(root)

    palm = stage.GetPrimAtPath('/G1/right_palm_link')
    assert palm and palm.HasAPI(UsdPhysics.RigidBodyAPI), 'Missing right_palm_link'

    offset = Gf.Vec3d(0.20, 0.0, 0.0)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    relative = cache.GetLocalToWorldTransform(palm) * cache.GetLocalToWorldTransform(root).GetInverse()
    paddle = UsdGeom.Xform.Define(stage, '/G1/g1_paddle_link')
    paddle.AddTransformOp().Set(Gf.Matrix4d().SetTranslate(offset) * relative)
    rigid = UsdPhysics.RigidBodyAPI.Apply(paddle.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)

    radius, thickness = 0.075, 0.010
    handle_radius, handle_length, handle_x = 0.013, 0.10, -0.11
    head_mass, handle_mass = 0.12, 0.04
    mass = head_mass + handle_mass
    com_x = handle_mass * handle_x / mass
    head_radial = head_mass * (3 * radius**2 + thickness**2) / 12
    handle_radial = handle_mass * (3 * handle_radius**2 + handle_length**2) / 12
    parallel = head_mass * com_x**2 + handle_mass * (handle_x - com_x)**2
    inertia = Gf.Vec3f(
        head_radial + 0.5 * handle_mass * handle_radius**2,
        0.5 * head_mass * radius**2 + handle_radial + parallel,
        head_radial + handle_radial + parallel,
    )
    mass_api = UsdPhysics.MassAPI.Apply(paddle.GetPrim())
    mass_api.CreateMassAttr(mass)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(com_x, 0, 0))
    mass_api.CreateDiagonalInertiaAttr(inertia)
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0))

    material = UsdShade.Material.Define(stage, '/G1/PaddlePhysicsMaterial')
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(0.5)
    physics_material.CreateDynamicFrictionAttr(0.5)
    physics_material.CreateRestitutionAttr(0.8)
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr('min')
    physx_material.CreateRestitutionCombineModeAttr('min')

    for name, axis, shape_radius, height, position, color in (
        ('Head', 'Y', radius, thickness, (0, 0, 0), (0.75, 0.05, 0.03)),
        ('Handle', 'X', handle_radius, handle_length, (handle_x, 0, 0), (0.25, 0.15, 0.08)),
    ):
        shape = UsdGeom.Cylinder.Define(stage, '/G1/g1_paddle_link/' + name)
        shape.CreateAxisAttr(axis)
        shape.CreateRadiusAttr(shape_radius)
        shape.CreateHeightAttr(height)
        shape.AddTranslateOp().Set(Gf.Vec3d(*position))
        shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        extent = ([(-shape_radius, -height/2, -shape_radius), (shape_radius, height/2, shape_radius)]
                  if axis == 'Y' else [(-height/2, -shape_radius, -shape_radius), (height/2, shape_radius, shape_radius)])
        shape.CreateExtentAttr([Gf.Vec3f(*v) for v in extent])
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
        collision.CreateContactOffsetAttr(0.001)
        collision.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material, materialPurpose='physics')

    fixed = UsdPhysics.FixedJoint.Define(stage, '/G1/g1_paddle_fixed_joint')
    fixed.CreateBody0Rel().SetTargets([palm.GetPath()])
    fixed.CreateBody1Rel().SetTargets([paddle.GetPath()])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(*offset))
    fixed.CreateLocalRot0Attr(Gf.Quatf(1.0))
    fixed.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    fixed.CreateLocalRot1Attr(Gf.Quatf(1.0))
    fixed.CreateJointEnabledAttr(True)
    fixed.CreateCollisionEnabledAttr(False)
    fixed.CreateExcludeFromArticulationAttr(False)
    assert stage.GetRootLayer().Export(str(output))
    print('[PASS] Paddle USD generated with rigid body, mass, colliders and fixed joint', flush=True)

    cfg_path = asset_dir / 'g1_tt.py'
    config_source = '''"""G1 table-tennis asset with a fixed right-hand paddle."""
from pathlib import Path
from isaaclab_assets import G1_CFG

G1_PADDLE_PARENT_BODY_NAME = "right_palm_link"
G1_PADDLE_BODY_NAME = "g1_paddle_link"
G1_PADDLE_PARENT_OFFSET = (0.20, 0.0, 0.0)
G1_PADDLE_CENTER_OFFSET = (0.0, 0.0, 0.0)
G1_PADDLE_NORMAL = (0.0, 1.0, 0.0)
G1_PADDLE_RADIUS = 0.075
G1_PADDLE_HALF_THICKNESS = 0.005

G1_TT_CFG = G1_CFG.copy()
G1_TT_CFG.spawn.usd_path = str(Path(__file__).resolve().parent / "G1_TT" / "G1_TT.usda")
G1_TT_CFG.init_state.pos = (-1.6, 0.0, 0.74)
G1_TT_CFG.spawn.activate_contact_sensors = True
'''
    ast.parse(config_source)
    if cfg_path.exists():
        shutil.copy2(cfg_path, cfg_path.with_name(cfg_path.name + '.' + stamp + '.bak'))
    cfg_path.write_text(config_source)

    details = dict(parent_body='right_palm_link', paddle_body='g1_paddle_link',
        parent_offset=list(offset), normal=[0, 1, 0], radius=radius, thickness=thickness,
        mass=mass, source_usd=G1_CFG.spawn.usd_path)
    (usd_dir / 'paddle_parameters.json').write_text(json.dumps(details, indent=2))
    print('[PASS] G1_TT_CFG now references the local paddle asset', flush=True)
    print('USD:', output, flush=True)
    print('STEP 3B PADDLE BUILD PASSED', flush=True)

    import importlib
    importlib.invalidate_caches()
    sys.modules.pop("legged_lab.assets.unitree.g1_tt", None)

    import omni.usd
    import torch
    from pxr import UsdPhysics
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.utils.math import quat_apply, quat_rotate_inverse
    from legged_lab.assets.unitree.g1_tt import (
        G1_TT_CFG, G1_PADDLE_BODY_NAME, G1_PADDLE_PARENT_BODY_NAME,
        G1_PADDLE_PARENT_OFFSET, G1_PADDLE_RADIUS, G1_PADDLE_NORMAL,
    )
    from legged_lab.assets.table_tennis.ball import BALL_CFG
    from legged_lab.envs.g1_tt.g1_tt_config import G1_CONTROL_JOINTS, G1_HAND_JOINTS

    assert Path(G1_TT_CFG.spawn.usd_path).is_file()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=0.002, device='cuda:0', physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode='min', restitution_combine_mode='min', restitution=0.8)))
    ground = sim_utils.GroundPlaneCfg()
    ground.func('/World/Ground', ground)
    robot = Articulation(G1_TT_CFG.replace(prim_path='/World/G1'))
    ball_cfg = BALL_CFG.copy()
    ball_cfg.prim_path = '/World/TestBall'
    ball_cfg.init_state.pos = (3.0, 0.0, 1.0)
    ball_cfg.spawn.rigid_props.disable_gravity = True
    ball = RigidObject(ball_cfg)
    sim.reset()

    assert set(robot.joint_names) == set(G1_CONTROL_JOINTS + G1_HAND_JOINTS)
    ids, names = robot.find_joints(G1_CONTROL_JOINTS, preserve_order=True)
    assert len(ids) == 23 and list(names) == G1_CONTROL_JOINTS
    palm_ids, _ = robot.find_bodies(G1_PADDLE_PARENT_BODY_NAME)
    paddle_ids, _ = robot.find_bodies(G1_PADDLE_BODY_NAME)
    assert len(palm_ids) == len(paddle_ids) == 1
    palm_id, paddle_id = palm_ids[0], paddle_ids[0]
    head = omni.usd.get_context().get_stage().GetPrimAtPath('/World/G1/g1_paddle_link/Head')
    assert head and head.HasAPI(UsdPhysics.CollisionAPI)
    print('[PASS] Paddle loaded; 37 joints and 23 control joints preserved', flush=True)

    state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(state[:, :7])
    robot.write_root_velocity_to_sim(state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()
    targets = robot.data.default_joint_pos.clone()
    wrist_ids, _ = robot.find_joints('right_elbow_roll_joint')
    wrist_id = wrist_ids[0]
    targets[:, wrist_id] += 0.15
    expected_offset = torch.tensor(G1_PADDLE_PARENT_OFFSET, device=sim.device).unsqueeze(0)
    normal_local = torch.tensor(G1_PADDLE_NORMAL, device=sim.device).unsqueeze(0)
    max_error = 0.0

    def step():
        global max_error
        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        ball.update(sim.get_physics_dt())
        assert torch.isfinite(robot.data.body_pos_w).all()
        palm_q = robot.data.body_quat_w[:, palm_id]
        paddle_q = robot.data.body_quat_w[:, paddle_id]
        local = quat_rotate_inverse(palm_q,
            robot.data.body_pos_w[:, paddle_id] - robot.data.body_pos_w[:, palm_id])
        max_error = max(max_error, float((local - expected_offset).norm()))
        assert float((palm_q * paddle_q).sum(-1).abs().min()) > 0.995

    for _ in range(100):
        step()
    movement = float((robot.data.joint_pos[:, wrist_id] - robot.data.default_joint_pos[:, wrist_id]).abs().max())
    assert movement > 0.01, ('Arm did not move', movement)
    assert max_error < 0.005, ('Fixed attachment error', max_error)
    print('[PASS] Paddle follows the moving palm; max error %.6f m' % max_error, flush=True)

    center = robot.data.body_pos_w[:, paddle_id].clone()
    normal = quat_apply(robot.data.body_quat_w[:, paddle_id], normal_local)
    state = ball.data.default_root_state.clone()
    state[:, :3] = center + 0.12 * normal
    state[:, 7:10] = -2.0 * normal + robot.data.body_lin_vel_w[:, paddle_id]
    state[:, 10:13] = 0.0
    ball.write_root_pose_to_sim(state[:, :7])
    ball.write_root_velocity_to_sim(state[:, 7:])
    ball.reset()
    bounced = False
    outgoing = float('-inf')
    for _ in range(80):
        step()
        normal = quat_apply(robot.data.body_quat_w[:, paddle_id], normal_local)
        relative = ball.data.root_pos_w - robot.data.body_pos_w[:, paddle_id]
        plane_distance = (relative * normal).sum(-1)
        radial = (relative - plane_distance.unsqueeze(-1) * normal).norm(dim=-1)
        relative_velocity = ball.data.root_lin_vel_w - robot.data.body_lin_vel_w[:, paddle_id]
        vn = float((relative_velocity * normal).sum(-1).item())
        if vn > 0.2 and abs(float(plane_distance.item())) < 0.04 and float(radial.item()) < G1_PADDLE_RADIUS:
            bounced = True
            outgoing = vn
            break
    assert bounced, 'No verified paddle-face rebound; inspect the scene before proceeding'
    assert max_error < 0.005, ('Fixed attachment error', max_error)
    assert BALL_CFG.spawn.rigid_props.disable_gravity is False
    print('[PASS] Physical ball rebound at paddle face: outgoing %.3f m/s' % outgoing, flush=True)
    print('STEP 3 PADDLE CHECK PASSED', flush=True)
finally:
    app.close()

