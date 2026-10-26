"""Reuse validated arm screening; full PACE owns all non-arm channels."""
import torch
from batch_geometry import BatchGeometry, quaternion_matrix
from target_guard import choose_targets
from runtime import bind_guard
from scene_audit import verify_clones
from gravity_support import RightArmGravitySupport
from ready_pose import RIGHT_READY_RAD
from contact_pairs import ContactPairs


class Safety:
    def __init__(self, env, fk, audit):
        self.e = env
        guard, self.views = bind_guard(env, fk, audit)
        verify_clones(env, guard, audit)
        make = lambda x: torch.as_tensor(x, dtype=torch.float32, device=env.device)
        self.geom = BatchGeometry(guard.fk, guard.bounds, guard.watch, torch, make)
        self.body_ids = [env.robot.body_names.index(n) for n in self.geom.names]
        self.arms = guard.arms
        self.checked = self.arms+[env.robot.joint_names.index('torso_joint')]
        self.rate = make([.9 if 'wrist' in env.robot.joint_names[j] else .6 for j in self.arms])
        self.hard, self.inner, self.soft = make([.01,.03,.02]), make([.015,.05,.035]), make([.04,.09,.06])
        self.last = env.ready_q.clone()
        self.contacts = ContactPairs(env, guard.watch)
        self.minimum = self.actual()
        assert (self.minimum >= self.hard).all(), 'READY INITIAL GEOMETRY FAILURE'
        assert torch.allclose(self.minimum, self.geom.inspect(env.robot.data.joint_pos), atol=.002, rtol=0.)
        self.gravity = RightArmGravitySupport(env, guard.fk, RIGHT_READY_RAD)
        self.gravity.apply(self.last)
        self.initial = {key: getattr(env.robot.root_physx_view, method)().clone() for key, method in
                        [('kp','get_dof_stiffnesses'), ('kd','get_dof_dampings'), ('limits','get_dof_max_forces')]}
        self.defaults = env.robot.data.default_joint_pos.clone()
        self.audit = audit

    def actual(self):
        d = self.e.robot.data
        return self.geom.measure(d.body_link_pos_w[:, self.body_ids],
                                 quaternion_matrix(d.body_link_quat_w[:, self.body_ids], torch))

    def select(self, wanted):
        e, d = self.e, self.e.robot.data
        wanted = wanted.clone()
        wanted[:, self.arms] = self.last[:, self.arms]+(wanted[:, self.arms]-self.last[:, self.arms]).clamp(-self.rate*e.step_dt, self.rate*e.step_dt)
        wanted = torch.maximum(torch.minimum(wanted, d.joint_pos_limits[:,:,1]), d.joint_pos_limits[:,:,0])
        selected, found, rows = [], [], []
        # Bound temporary FK/SAT memory without reducing sampling density.
        for start in range(0, e.num_envs, 128):
            end = min(start+128, e.num_envs)
            q, v = d.joint_pos[start:end], d.joint_vel[start:end]
            target, ok, info = choose_targets(q, v, self.last[start:end], wanted[start:end],
                self.arms, self.checked, self.rate, e.step_dt, self.geom, self.inner,
                .08, torch, lambda a: q.new_tensor(a), self.hard, e.ready_q[start:end])
            selected.append(target)
            found.append(ok)
            # The guard already reduces paths to fixed candidate arrays; retain
            # reasons and pre-physics snapshots for any later failure analysis.
            rows.append({k: val.detach().clone() for k,val in info.items()})
        self.info = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
        target, ok = torch.cat(selected), torch.cat(found)
        self.no_path = ~ok
        self.last = target.clone()
        self.minimum = self.actual()
        return target, ~ok

    def reset(self, ids):
        self.last[ids] = self.e.ready_q[ids]
        self.gravity.apply(self.e.robot.data.joint_pos_target, ids)

    def verify(self):
        e = self.e
        self.gravity.verify_unchanged()
        assert torch.equal(e.robot.data.default_joint_pos, self.defaults)
        for key, method in [('kp','get_dof_stiffnesses'), ('kd','get_dof_dampings'), ('limits','get_dof_max_forces')]:
            assert torch.equal(getattr(e.robot.root_physx_view, method)(), self.initial[key]), key+' changed'
