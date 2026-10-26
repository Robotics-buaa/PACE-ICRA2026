"""Ephemeral USD pair filters and a geometry-validated arm safety adapter."""
import hashlib
from itertools import combinations
from pathlib import Path
import numpy as np
from guard import ArmGuard, FK, protected_links, quat_matrix, transform
from collider_completion import complete_missing, verify_physics_mass
from ready_pose import SOURCE_URDF_SHA256

URDF = Path('/workspace/projects/unitree_ros_h1_2_assets/robots/h1_2_description/h1_2_handless.urdf')


def configure_self_contacts(cfg, action_names):
    """Run after AppLauncher, BEFORE spawning/physics initialization. No USD save."""
    from pxr import Usd, UsdPhysics, PhysxSchema
    import omni.usd
    xml = URDF.read_text()
    assert hashlib.sha256(URDF.read_bytes()).hexdigest() == SOURCE_URDF_SHA256, 'READY CANDIDATE SOURCE URDF CHANGED'
    fk = FK(xml, action_names)
    watch = protected_links(fk)
    audit = {'urdf': str(URDF), 'urdf_sha256': hashlib.sha256(URDF.read_bytes()).hexdigest(),
             'scope': 'ONLY upperarm/elbow/wrists/paddle vs torso/pelvis; other self pairs retain disabled behavior',
             'usd_changes': 'session layer of newly spawned scene only; no asset/layer saves'}
    protected = {frozenset((arm, body)) for arm in watch for body in ('torso_link', 'pelvis')}
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
    original_spawn = cfg.scene.robot.spawn.func
    def spawn(prim_path, spawn_cfg, *args, **kwargs):
        prim = original_spawn(prim_path, spawn_cfg, *args, **kwargs)
        stage = omni.usd.get_context().get_stage()
        root_path = prim.GetPath()
        descendants = list(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
        bodies = {p.GetName(): p for p in descendants if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        assert set(watch+['torso_link', 'pelvis']) <= bodies.keys(), 'PROTECTED RIGID BODIES MISSING'
        assert len(bodies) == len([p for p in descendants if p.HasAPI(UsdPhysics.RigidBodyAPI)])
        audit['collider_completion'] = complete_missing(stage, bodies, watch+['torso_link', 'pelvis'])
        descendants = list(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
        roots = [p for p in descendants if p.HasAPI(PhysxSchema.PhysxArticulationAPI)]
        assert roots and all(PhysxSchema.PhysxArticulationAPI(p).GetEnabledSelfCollisionsAttr().Get() for p in roots), 'SELF COLLISIONS NOT ENABLED'
        removed, exclusions = 0, 0
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for p in descendants:
                if p.HasAPI(UsdPhysics.FilteredPairsAPI):
                    relation = UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel()
                    targets = relation.GetTargets()
                    kept = [target for target in targets if not target.HasPrefix(root_path)]
                    if len(kept) != len(targets):
                        assert not p.IsInstanceProxy(), 'CANNOT OVERRIDE INTERNAL INSTANCE COLLISION FILTER'
                        relation.SetTargets(kept)
                        removed += len(targets)-len(kept)
            for a, b in combinations(sorted(bodies), 2):
                if frozenset((a, b)) not in protected:
                    UsdPhysics.FilteredPairsAPI.Apply(bodies[a]).CreateFilteredPairsRel().AddTarget(bodies[b].GetPath())
                    exclusions += 1
        # Verify every surviving internal body/collider/articulation relationship.
        def owner(path):
            for name, body in bodies.items():
                if path.HasPrefix(body.GetPath()):
                    return name
            return None
        for p in descendants:
            if p.HasAPI(UsdPhysics.FilteredPairsAPI):
                for target in UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets():
                    if target.HasPrefix(root_path):
                        a, b = owner(p.GetPath()), owner(target)
                        assert a is not None and b is not None and frozenset((a, b)) not in protected, 'PROTECTED CONTACT STILL FILTERED'
        audit.update(self_collisions_enabled=True, protected_pairs=[sorted(p) for p in sorted(protected, key=lambda x: sorted(x))],
                     removed_internal_filter_targets=removed, excluded_other_body_pairs=exclusions)
        print('[FRONT-GUARD] SELF_CONTACT_SETUP:', audit, flush=True)
        return prim
    cfg.scene.robot.spawn.func = spawn
    return fk, audit


def bind_guard(env, fk, audit):
    from pxr import Usd, UsdGeom, UsdPhysics
    import omni.usd
    from legged_lab.assets.unitree.h1_2_tt import H1_2_PADDLE_NORMAL
    audit['completed_link_physics_mass'] = verify_physics_mass(env, audit['collider_completion'])
    stage = omni.usd.get_context().get_stage()
    root = env.scene.env_prim_paths[0]+'/Robot/'
    names = list(env.robot.joint_names)
    assert set(names) == set(fk.names), 'FK JOINT SET MISMATCH'
    fk = FK(URDF.read_text(), names)  # PhysX order can differ from policy action order.
    q = env.robot.data.joint_pos[0].detach().cpu().numpy().astype(float)
    body_names = list(env.robot.body_names)
    p = env.robot.data.body_link_pos_w[0].detach().cpu().numpy()
    r = env.robot.data.body_link_quat_w[0].detach().cpu().numpy()
    base = body_names.index('pelvis')
    to_base = np.linalg.inv(transform(p[base], quat_matrix(r[base])))
    actual = {name: to_base@transform(p[i], quat_matrix(r[i])) for i, name in enumerate(body_names)}
    predicted = fk.forward(q)
    fk.paddle_mount = np.linalg.inv(predicted['right_wrist_yaw_link'])@actual['h1_2_paddle_link']
    predicted = fk.forward(q)
    watched = protected_links(fk)+['torso_link', 'pelvis']
    errors = {name: {'position_m': float(np.linalg.norm(predicted[name][:3, 3]-actual[name][:3, 3])),
                     'rotation_matrix_max_error': float(np.max(np.abs(predicted[name][:3, :3]-actual[name][:3, :3])))} for name in watched}
    assert all(v['position_m'] < .002 and v['rotation_matrix_max_error'] < .005 for v in errors.values()), 'URDF/USD FK MISMATCH: '+str(errors)
    audit['fk_validation'] = errors
    bounds, collider_paths = {}, {}
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)
    for name in watched:
        body = stage.GetPrimAtPath(root+name)
        boxes, paths = [], []
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
                bound = cache.ComputeRelativeBound(prim, body).ComputeAlignedRange()
                if not bound.IsEmpty():
                    boxes.append(np.array([list(bound.GetMin()), list(bound.GetMax())]))
                    paths.append(str(prim.GetPath()))
        assert boxes, 'NO ENABLED COLLIDER: '+name
        bounds[name] = np.array([np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)])
        assert np.isfinite(bounds[name]).all() and (bounds[name][1] > bounds[name][0]).all()
        collider_paths[name] = paths
    audit.update(collider_paths=collider_paths, local_collision_bounds_m={name: box.tolist() for name, box in bounds.items()})
    limits = env.robot.data.joint_pos_limits[0].detach().cpu().numpy()
    guard = ArmGuard(fk, bounds, limits, q, env.step_dt)
    ready = guard.prepare(q, H1_2_PADDLE_NORMAL)
    audit.update(fk_validation=errors, collider_paths=collider_paths,
                 local_collision_bounds_m={name: box.tolist() for name, box in bounds.items()}, preparation=ready)
    # Each source body is filtered only against torso/pelvis. Sum over any
    # collision-shape expansion of the two filter patterns; never use ball contacts.
    pair_views = [env.contact_sensor._physics_sim_view.create_rigid_contact_view(root+name,
                  filter_patterns=[root+'torso_link', root+'pelvis']) for name in guard.watch]
    assert all(view.filter_count >= 2 for view in pair_views), 'SELF CONTACT FILTER BINDING FAILED'
    print('[FRONT-GUARD] VERIFIED_FRONT_PREPARATION:', ready, flush=True)
    return guard, pair_views


def apply_guard(env, actions, guard, time_s):
    """Filter14 arm targets, preserving13 leg/torso action channels bit-for-bit."""
    import torch
    action_ids = [int(j) for j in env.action_joint_ids]
    actual = env.robot.data.joint_pos[0].detach().cpu().numpy().astype(float)
    defaults = env.robot.data.default_joint_pos[0, action_ids]
    targets = torch.clip(actions, -env.clip_actions, env.clip_actions)*env.action_scale+defaults
    requested = actual.copy()
    requested[action_ids] = targets[0].detach().cpu().numpy()
    result = guard.select(actual, requested, time_s)
    applied = actions.clone()
    # action_scale can be scalar or per-action; derive equivalent full residuals.
    target_tensor = actions.new_tensor(result[action_ids]).reshape(1, -1)
    equivalent = (target_tensor-defaults)/env.action_scale
    for i, joint in enumerate(action_ids):
        if joint in guard.arms:
            applied[:, i] = equivalent[:, i]
    return applied


def verify_urdf(audit):
    assert hashlib.sha256(URDF.read_bytes()).hexdigest() == audit['urdf_sha256'], 'SOURCE URDF CHANGED'
