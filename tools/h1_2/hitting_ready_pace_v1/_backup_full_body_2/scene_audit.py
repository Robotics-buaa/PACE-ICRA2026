"""Startup-only clone audit. Reject missing geometry, pair remaps or mass drift."""
import numpy as np
import torch


def verify_clones(env, guard, audit):
    from pxr import Usd, UsdPhysics, PhysxSchema
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    expected = {frozenset((arm, body)) for arm in guard.watch for body in ('torso_link', 'pelvis')}
    # Existing front runtime modifies the source before homogeneous cloning.
    # Verify the resulting stage, rather than assuming every clone inherited it.
    assert env.scene.cfg.replicate_physics, 'EXPECTED HOMOGENEOUS SOURCE-ENV CLONING'
    checked = 0
    for env_path in env.scene.env_prim_paths:
        root = stage.GetPrimAtPath(env_path+'/Robot')
        assert root.IsValid()
        descendants = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
        bodies = {p.GetName(): p for p in descendants if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        assert set(guard.watch+['torso_link', 'pelvis']) <= set(bodies)
        articulation = [p for p in descendants if p.HasAPI(PhysxSchema.PhysxArticulationAPI)]
        assert articulation and all(PhysxSchema.PhysxArticulationAPI(p).GetEnabledSelfCollisionsAttr().Get() for p in articulation)
        for name in guard.watch+['torso_link', 'pelvis']:
            collider = [p for p in Usd.PrimRange(bodies[name], Usd.TraverseInstanceProxies())
                        if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
            assert collider, 'CLONE MISSING PROTECTED COLLIDER: '+env_path+'/'+name
        def owner(path):
            for name, body in bodies.items():
                if path.HasPrefix(body.GetPath()):
                    return name
            return None
        for p in descendants:
            if p.HasAPI(UsdPhysics.FilteredPairsAPI):
                for target in UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets():
                    if str(target).startswith('/World/envs/'):
                        assert str(target).startswith(env_path+'/'), 'CLONED FILTER POINTS TO ANOTHER ENV'
                    if target.HasPrefix(root.GetPath()):
                        pair = frozenset((owner(p.GetPath()), owner(target)))
                        assert None not in pair and pair not in expected, 'PROTECTED CLONE PAIR STILL FILTERED'
        checked += 1
    view = env.robot.root_physx_view
    masses = view.get_masses().to(env.device)
    coms = view.get_coms().to(env.device)
    tensors = view.get_inertias().to(env.device).reshape(env.num_envs, -1, 3, 3)
    for name, data in audit['collider_completion']['added'].items():
        j = env.robot.body_names.index(name)
        ref = data['preserved_mass_properties']
        assert torch.allclose(masses[:, j], masses.new_full((env.num_envs,), ref['mass_kg']), atol=1e-6, rtol=1e-4)
        assert torch.allclose(coms[:, j, :3], coms.new_tensor(ref['center_of_mass_m']).expand(env.num_envs, -1), atol=1e-5, rtol=0.)
        eigen = torch.linalg.eigvalsh(tensors[:, j])
        assert torch.allclose(eigen, eigen.new_tensor(sorted(ref['diagonal_inertia_kgm2'])).expand(env.num_envs, -1), atol=1e-8, rtol=.001)
    audit['cloned_env_count_verified'] = checked
