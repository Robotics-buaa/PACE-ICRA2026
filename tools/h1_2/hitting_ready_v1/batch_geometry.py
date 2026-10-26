"""Batched FK and 22-pair SAT. numpy backend for math tests; torch for rollout.

Only construction consumes CPU geometry. No loop over environments in rollout.
SAT is a conservative box screen, not a continuous-time collision guarantee.
"""
import numpy as np


def quaternion_matrix(q, xp):
    q = q / xp.sqrt((q*q).sum(-1)[..., None] + 1e-20)
    w, x, y, z = [q[..., i] for i in range(4)]
    return xp.stack((
        xp.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)), -1),
        xp.stack((2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)), -1),
        xp.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1)), -2)


class BatchGeometry:
    def __init__(self, fk, bounds, watch, xp, array):
        self.xp, self.a = xp, array
        self.names = list(bounds)
        self.watch = list(watch)
        self.bounds = bounds
        self.centers = array(np.stack([(bounds[n][0]+bounds[n][1])*.5 for n in self.names]))
        self.extents = array(np.stack([(bounds[n][1]-bounds[n][0])*.5 for n in self.names]))
        self.ia = [self.names.index(n) for n in watch for b in ('torso_link', 'pelvis')]
        self.ib = [self.names.index(b) for n in watch for b in ('torso_link', 'pelvis')]
        self.torso = self.names.index('torso_link')
        self.paddle = self.names.index('h1_2_paddle_link')
        self.wrist = self.names.index('right_wrist_yaw_link')
        self.torso_front = float(bounds['torso_link'][1, 0])
        self.root = fk.root
        self.joints = []
        for parent, child, name, kind, origin, axis in fk.joints:
            v = np.asarray(axis if kind != 'fixed' else [1., 0., 0.], dtype=float)
            v /= np.linalg.norm(v)
            x, y, z = v
            k = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
            self.joints.append((parent, child, None if kind == 'fixed' else fk.index[name],
                                array(origin), array(k), array(k@k)))
        self.eye = array(np.eye(3))
        self.mount = array(fk.paddle_mount)

    def forward(self, q):
        xp = self.xp
        # Homogeneous transforms, with arbitrary leading batch dimensions.
        base = self.a(np.eye(4)) + q[..., :1, None]*0.
        poses = {self.root: base}
        for parent, child, index, origin, k, k2 in self.joints:
            t = poses[parent] @ origin
            if index is not None:
                angle = q[..., index, None, None]
                r = self.eye + xp.sin(angle)*k + (1.-xp.cos(angle))*k2
                top = xp.concatenate((t[..., :3, :3]@r, t[..., :3, 3:4]), -1)
                t = xp.concatenate((top, t[..., 3:4, :]), -2)
            poses[child] = t
        poses['h1_2_paddle_link'] = poses['right_wrist_yaw_link'] @ self.mount
        p = xp.stack([poses[n][..., :3, 3] for n in self.names], -2)
        r = xp.stack([poses[n][..., :3, :3] for n in self.names], -3)
        return p, r

    def measure(self, p, r):
        xp = self.xp
        c = p + (r @ self.centers[..., None])[..., 0]
        ca, cb = c[..., self.ia, :], c[..., self.ib, :]
        ra, rb = r[..., self.ia, :, :], r[..., self.ib, :, :]
        aa, ab = xp.swapaxes(ra, -1, -2), xp.swapaxes(rb, -1, -2)
        cross = xp.cross(aa[..., :, None, :], ab[..., None, :, :], dim=-1) if xp.__name__ == 'torch' else xp.cross(aa[..., :, None, :], ab[..., None, :, :])
        cross = cross.reshape(*cross.shape[:-3], 9, 3)
        axes = xp.concatenate((aa, ab, cross), -2)
        norm = xp.sqrt((axes*axes).sum(-1)[..., None])
        axes = axes / xp.maximum(norm, self.a(1e-8))
        distance = xp.abs((axes*(ca-cb)[..., None, :]).sum(-1))
        rad_a = (xp.abs(axes@ra)*self.extents[self.ia][..., None, :]).sum(-1)
        rad_b = (xp.abs(axes@rb)*self.extents[self.ib][..., None, :]).sum(-1)
        gap = xp.where(norm[..., 0] > 1e-8, distance-rad_a-rad_b, self.a(-1e6))
        smallest = xp.amin(xp.amax(gap, -1), -1)
        tr = xp.swapaxes(r[..., self.torso, :, :], -1, -2)
        center_t = (tr @ (c[..., self.paddle, :]-p[..., self.torso, :])[..., None])[..., 0]
        relative_r = tr @ r[..., self.paddle, :, :]
        front = center_t[..., 0] - (xp.abs(relative_r[..., 0, :])*self.extents[self.paddle]).sum(-1)-self.torso_front
        wrist = (tr @ (p[..., self.wrist, :]-p[..., self.torso, :])[..., None])[..., 0, 0]-self.torso_front
        return xp.stack((smallest, front, wrist), -1)

    def inspect(self, q):
        return self.measure(*self.forward(q))
