"""Right-arm gravity feedforward from live masses, COMs and joint axes.

Only seven right-arm torques are written. PD gains, joint targets, force limits,
root gravity and the frozen stance controller are unchanged. This is gravity
support, not full inverse dynamics or a claim of measured actuator effort.
"""
import numpy as np
from batch_geometry import quaternion_matrix


def gravity_moments(position, rotation, masses, local_com, children, axes,
                    descendant_mask, gravity, xp):
    center = position+(rotation@local_com[..., None])[..., 0]
    origin = position[:, children]
    axis_world = (rotation[:, children]@axes[..., None])[..., 0]
    lever = center[:, None]-origin[:, :, None]
    force = masses[..., None]*gravity
    if xp.__name__ == 'torch':
        moment = xp.linalg.cross(lever, force[:, None].expand_as(lever), dim=-1)
    else:
        moment = xp.cross(lever, force[:, None])
    # External gravity moment is a . (r x mg). Its negative is required support.
    return -(moment*axis_world[:, :, None]*descendant_mask[None, :, :, None]).sum(-1).sum(-1)


def bounded_support(raw, pd, limits, xp):
    cap = .5*limits
    low = xp.maximum(-cap, -limits-pd)
    high = xp.minimum(cap, limits-pd)
    feasible = xp.all(low <= high, -1)
    applied = xp.maximum(xp.minimum(raw, high), low)
    return applied, feasible, xp.abs(applied-raw) > 1e-5


def topology(fk, body_names, joint_names):
    child_map, axis_map, graph = {}, {}, {}
    for parent, child, joint, kind, origin, axis in fk.joints:
        graph.setdefault(parent, set()).add(child)
        child_map[joint] = child
        axis_map[joint] = np.asarray(axis, dtype=float)
    graph.setdefault('right_wrist_yaw_link', set()).add('h1_2_paddle_link')
    children, axes, masks = [], [], []
    for joint in joint_names:
        child = child_map[joint]
        reachable, pending = set(), [child]
        while pending:
            node = pending.pop()
            if node not in reachable:
                reachable.add(node)
                pending.extend(graph.get(node, ()))
        children.append(body_names.index(child))
        axis = axis_map[joint]
        assert np.linalg.norm(axis) > .99
        axes.append(axis/np.linalg.norm(axis))
        masks.append([float(n in reachable) for n in body_names])
    return children, np.asarray(axes), np.asarray(masks)


class RightArmGravitySupport:
    def __init__(self, env, fk, right_names):
        import torch
        self.torch, self.env, self.fk = torch, env, fk
        self.right_names = list(right_names)
        self.right = [env.robot.joint_names.index(n) for n in right_names]
        self.names = list(env.robot.body_names)
        self.children, axes, masks = topology(fk, self.names, right_names)
        make = lambda a: torch.as_tensor(a, dtype=torch.float32, device=env.device)
        self.axes, self.masks = make(axes), make(masks)
        view = env.robot.root_physx_view
        self.masses = view.get_masses().to(env.device).clone()
        self.local_com = view.get_coms().to(env.device)[:, :, :3].clone()
        self.gravity = make(env.sim.cfg.gravity)
        assert self.gravity.shape == (3,) and torch.isfinite(self.gravity).all()
        assert self.gravity.norm() > 1., 'GRAVITY MUST REMAIN ENABLED'
        assert self.masses.shape == (env.num_envs, len(self.names))
        assert self.local_com.shape == (env.num_envs, len(self.names), 3)
        assert torch.isfinite(self.masses).all() and (self.masses > 0).all()
        assert torch.isfinite(self.local_com).all()
        self.limits = env.robot.data.joint_effort_limits[:, self.right].clone()
        assert (self.limits > 0).all()
        self.last = {}
        self.audit = self.validate_initial()

    def raw(self):
        d, torch = self.env.robot.data, self.torch
        return gravity_moments(d.body_link_pos_w, quaternion_matrix(d.body_link_quat_w, torch),
                               self.masses, self.local_com, self.children, self.axes, self.masks,
                               self.gravity, torch)

    def validate_initial(self):
        from guard import transform, quat_matrix
        e, torch = self.env, self.torch
        q = e.robot.data.joint_pos[0].detach().cpu().numpy().astype(float)
        positions = e.robot.data.body_link_pos_w[0].detach().cpu().numpy().astype(float)
        quats = e.robot.data.body_link_quat_w[0].detach().cpu().numpy().astype(float)
        base = self.names.index('pelvis')
        world_from_pelvis = transform(positions[base], quat_matrix(quats[base]))
        predicted = self.fk.forward(q)
        weighted = np.where(self.masks.detach().cpu().numpy().any(0))[0].tolist()
        errors = {}
        for j in weighted:
            name = self.names[j]
            assert name in predicted, 'GRAVITY BODY NOT IN VERIFIED FK: '+name
            actual = transform(positions[j], quat_matrix(quats[j]))
            expected = world_from_pelvis@predicted[name]
            errors[name] = dict(position_m=float(np.linalg.norm(actual[:3, 3]-expected[:3, 3])),
                                rotation_max=float(np.abs(actual[:3, :3]-expected[:3, :3]).max()))
        assert all(v['position_m'] < .002 and v['rotation_max'] < .005 for v in errors.values()), 'GRAVITY AXIS/COM FRAME VALIDATION FAILED'
        m = self.masses[0].detach().cpu().numpy().astype(float)
        com = self.local_com[0].detach().cpu().numpy().astype(float)
        g = self.gravity.detach().cpu().numpy().astype(float)
        def potential(a):
            poses = self.fk.forward(a)
            value = 0.
            for j in weighted:
                pose = world_from_pelvis@poses[self.names[j]]
                center = pose[:3, :3]@com[j]+pose[:3, 3]
                value -= m[j]*float(g@center)
            return value
        finite_difference = []
        for joint in self.right:
            plus, minus = q.copy(), q.copy()
            plus[joint] += 1e-5
            minus[joint] -= 1e-5
            finite_difference.append((potential(plus)-potential(minus))/2e-5)
        measured_model = self.raw()[0].detach().cpu().numpy()
        assert np.allclose(measured_model, finite_difference, atol=.05, rtol=.02), 'GRAVITY TORQUE SIGN/ORDER/FRAME MISMATCH'
        assert (np.abs(measured_model) <= .5*self.limits[0].cpu().numpy()).all(), (
            'READY GRAVITY SUPPORT EXCEEDS FEEDFORWARD CAP: required='+str(measured_model.tolist())
            +' cap='+str((.5*self.limits[0]).cpu().tolist()))
        audit = dict(method='negative world moment of live descendant-link gravity forces; right-arm only',
                     joint_names=self.right_names, live_model_Nm=measured_model.tolist(),
                     potential_gradient_Nm=list(map(float, finite_difference)),
                     max_FD_difference_Nm=float(np.max(np.abs(measured_model-finite_difference))),
                     feedforward_cap_fraction_of_existing_effort_limit=.5,
                     frame_errors=errors, is_inverse_dynamics=False)
        print('[READY-GRAVITY] VALIDATED:', audit, flush=True)
        return audit

    def apply(self, target, ids=None):
        e, torch = self.env, self.torch
        d = e.robot.data
        raw = self.raw()
        pd = (d.joint_stiffness[:, self.right]*(target[:, self.right]-d.joint_pos[:, self.right])
              +d.joint_damping[:, self.right]*(d.joint_vel_target[:, self.right]-d.joint_vel[:, self.right]))
        applied, feasible, clipped = bounded_support(raw, pd, self.limits, torch)
        assert torch.isfinite(raw).all() and torch.isfinite(applied).all(), 'NONFINITE GRAVITY SUPPORT'
        self.last = dict(gravity_raw_Nm=raw.detach().clone(), gravity_applied_Nm=applied.detach().clone(),
                         gravity_clipped=clipped.detach().clone(), gravity_pd_estimate_Nm=pd.detach().clone(),
                         gravity_estimated_total_ratio=((pd+applied).abs()/self.limits).detach().clone())
        assert (feasible if ids is None else feasible[ids]).all(), 'GRAVITY SUPPORT: NO EFFORT HEADROOM; physics not advanced'
        full = torch.zeros_like(d.joint_pos)
        full[:, self.right] = applied
        if ids is None:
            e.robot.set_joint_effort_target(full)
        else:
            e.robot.set_joint_effort_target(full[ids], env_ids=ids)
        return full

    def verify_unchanged(self):
        e, torch = self.env, self.torch
        view = e.robot.root_physx_view
        assert torch.equal(view.get_masses().to(e.device), self.masses), 'LIVE MASSES CHANGED'
        assert torch.equal(view.get_coms().to(e.device)[:, :, :3], self.local_com), 'LIVE COMS CHANGED'
        assert torch.equal(self.gravity, self.gravity.new_tensor(e.sim.cfg.gravity)), 'GRAVITY CHANGED'
