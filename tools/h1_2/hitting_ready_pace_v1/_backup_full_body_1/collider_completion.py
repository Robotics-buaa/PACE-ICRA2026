"""Complete missing protected-link collision geometry in the scene session only.

Keep authored mass properties; reject unknown inertia rather than recooking mass.
Existing usable colliders are never replaced. No original USD/mesh file writes.
"""
import math
import numpy as np


def proxy_box(bounds, padding=.001):
    bounds = np.asarray(bounds, dtype=float)
    assert bounds.shape == (2, 3) and np.isfinite(bounds).all(), 'INVALID VISUAL BOUNDS'
    size = bounds[1]-bounds[0]
    assert (size >= .0001).all() and (size <= 1.5).all(), 'IMPLAUSIBLE VISUAL SIZE: '+str(size)
    return (bounds[0]+bounds[1])*.5, size*.5+padding


def mass_properties(body, UsdPhysics):
    api = UsdPhysics.MassAPI(body)
    mass = api.GetMassAttr().Get()
    inertia = api.GetDiagonalInertiaAttr().Get()
    center = api.GetCenterOfMassAttr().Get()
    axes = api.GetPrincipalAxesAttr().Get()
    info = dict(mass_kg=None if mass is None else float(mass),
                diagonal_inertia_kgm2=None if inertia is None else list(map(float, inertia)),
                center_of_mass_m=None if center is None else list(map(float, center)),
                principal_axes_wxyz=None if axes is None else [float(axes.GetReal()), *map(float, axes.GetImaginary())])
    valid = (info['mass_kg'] is not None and info['mass_kg'] > 0.
             and all(info[key] is not None for key in ('diagonal_inertia_kgm2', 'center_of_mass_m', 'principal_axes_wxyz')))
    if valid:
        values = [info['mass_kg']]+info['diagonal_inertia_kgm2']+info['center_of_mass_m']+info['principal_axes_wxyz']
        valid = all(math.isfinite(v) for v in values) and min(info['diagonal_inertia_kgm2']) > 0.
        valid = valid and abs(sum(v*v for v in info['principal_axes_wxyz'])-1.) < .001
    assert valid, 'EXPLICIT MASS/INERTIA REQUIRED BEFORE COLLIDER COMPLETION: '+str(body.GetPath())+' '+str(info)
    return info


def complete_missing(stage, bodies, names):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)
    additions, retained = {}, {}
    # Plan ALL bodies first; a missing geometry/mass must fail before any box write.
    plans = []
    for name in names:
        body = bodies[name]
        descendants = list(Usd.PrimRange(body, Usd.TraverseInstanceProxies()))
        def belongs(p):
            while p and p != body:
                if p.HasAPI(UsdPhysics.RigidBodyAPI):
                    return False
                p = p.GetParent()
            return p == body
        descendants = [p for p in descendants if belongs(p)]
        enabled = [p for p in descendants if p.HasAPI(UsdPhysics.CollisionAPI)
                   and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
        if enabled:
            assert any(not cache.ComputeRelativeBound(p, body).ComputeAlignedRange().IsEmpty() for p in enabled), 'EXISTING COLLIDERS HAVE EMPTY BOUNDS: '+name
            retained[name] = [str(p.GetPath()) for p in enabled]
            continue
        visual_bounds, visual_paths = [], []
        for p in descendants:
            if p.IsA(UsdGeom.Boundable) and not p.HasAPI(UsdPhysics.CollisionAPI):
                bound = cache.ComputeRelativeBound(p, body).ComputeAlignedRange()
                if not bound.IsEmpty():
                    visual_bounds.append(np.array([list(bound.GetMin()), list(bound.GetMax())]))
                    visual_paths.append(str(p.GetPath()))
        assert visual_bounds, 'MISSING COLLIDER AND NO MEASURABLE VISUAL GEOMETRY: '+name
        bounds = np.array([np.min([b[0] for b in visual_bounds], axis=0), np.max([b[1] for b in visual_bounds], axis=0)])
        center, half = proxy_box(bounds)
        snapshot = mass_properties(body, UsdPhysics)
        path = str(body.GetPath())+'/front_guard_collision_box'
        assert not stage.GetPrimAtPath(path).IsValid(), 'UNEXPECTED COLLISION PROXY ALREADY EXISTS: '+path
        plans.append((name, body, path, center, half, snapshot, bounds, visual_paths))
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for name, body, path, center, half, snapshot, bounds, paths in plans:
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(2.)
            cube.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
            cube.AddScaleOp().Set(Gf.Vec3f(*map(float, half)))
            cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
            # Explicit positive mass/inertia on the parent body already override
            # shape-based automatic computation. Do not author density/new mass.
            if not body.HasAPI(PhysxSchema.PhysxContactReportAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.)
            assert mass_properties(body, UsdPhysics) == snapshot, 'AUTHORED MASS CHANGED: '+name
            additions[name] = dict(collider_path=path, geometry='invisible body-local visual-AABB cube, padded1mm',
                                   visual_paths=paths, visual_bounds_m=bounds.tolist(), center_m=center.tolist(),
                                   half_extents_m=half.tolist(), preserved_mass_properties=snapshot)
    result = dict(added=additions, retained_existing=retained,
                  scope='scene session only, before physics initialization; no asset saves, no mass/inertia authoring')
    print('[FRONT-GUARD] COLLIDER_COMPLETION:', result, flush=True)
    return result


def verify_physics_mass(env, completion):
    """Check the cooked body mass/COM/inertia spectrum against authored properties."""
    added = completion['added']
    if not added:
        return {}
    view = env.robot.root_physx_view
    masses = view.get_masses()[0].detach().cpu().numpy()
    centers = view.get_coms()[0, :, :3].detach().cpu().numpy()
    tensors = view.get_inertias()[0].detach().cpu().numpy().reshape(-1, 3, 3)
    names = list(env.robot.body_names)
    checked = {}
    for name, data in added.items():
        i = names.index(name)
        expected = data['preserved_mass_properties']
        actual_mass = float(masses[i])
        spectrum = np.linalg.eigvalsh(tensors[i])
        assert np.isclose(actual_mass, expected['mass_kg'], rtol=1e-4, atol=1e-6), 'COOKED MASS MISMATCH: '+name
        assert np.allclose(centers[i], expected['center_of_mass_m'], rtol=0., atol=1e-5), 'COOKED CENTER OF MASS MISMATCH: '+name
        assert np.allclose(spectrum, sorted(expected['diagonal_inertia_kgm2']), rtol=.001, atol=1e-8), 'COOKED INERTIA MISMATCH: '+name
        checked[name] = dict(mass_kg=actual_mass, center_of_mass_m=centers[i].tolist(), inertia_eigenvalues_kgm2=spectrum.tolist())
    print('[FRONT-GUARD] PRESERVED_BODY_MASS:', checked, flush=True)
    return checked
