"""URDF FK, conservative collision-box separation and bounded arm target filter.

No simulator state writes. Geometry/FK must be validated against live USD first.
"""
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

MARGIN_M = .01
FRONT_MARGIN_M = .03
ARMS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
        'wrist_roll', 'wrist_pitch', 'wrist_yaw')


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis/np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+math.sin(angle)*skew+(1.-math.cos(angle))*(skew@skew)


def transform(position=(0., 0., 0.), matrix=None):
    result = np.eye(4)
    result[:3, 3] = position
    if matrix is not None:
        result[:3, :3] = matrix
    return result


def quat_matrix(wxyz):
    w, x, y, z = np.asarray(wxyz)/np.linalg.norm(wxyz)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class FK:
    def __init__(self, xml, names):
        self.names = list(names)
        self.index = {n: i for i, n in enumerate(names)}
        tree = ET.fromstring(xml)
        pending = []
        children = set()
        parents = set()
        self.child_by_joint = {}
        for joint in tree.findall('joint'):
            name, kind = joint.attrib['name'], joint.attrib['type']
            parent = joint.find('parent').attrib['link']
            child = joint.find('child').attrib['link']
            origin = joint.find('origin')
            xyz = [float(v) for v in origin.attrib.get('xyz', '0 0 0').split()] if origin is not None else [0., 0., 0.]
            rpy = [float(v) for v in origin.attrib.get('rpy', '0 0 0').split()] if origin is not None else [0., 0., 0.]
            r = rotation([0, 0, 1], rpy[2])@rotation([0, 1, 0], rpy[1])@rotation([1, 0, 0], rpy[0])
            axis = joint.find('axis')
            axis = [float(v) for v in axis.attrib.get('xyz', '1 0 0').split()] if axis is not None else [1., 0., 0.]
            assert kind in ('fixed', 'revolute', 'continuous'), 'UNSUPPORTED URDF JOINT: '+name
            assert kind == 'fixed' or name in self.index, 'UNMAPPED URDF JOINT: '+name
            pending.append((parent, child, name, kind, transform(xyz, r), axis))
            children.add(child)
            parents.add(parent)
            self.child_by_joint[name] = child
        roots = parents-children
        assert len(roots) == 1
        self.root = roots.pop()
        self.joints = []
        known = {self.root}
        while pending:
            ready = [j for j in pending if j[0] in known]
            assert ready, 'DISCONNECTED URDF TREE'
            for joint in ready:
                self.joints.append(joint)
                known.add(joint[1])
                pending.remove(joint)
        assert 'pelvis' in known and 'torso_link' in known
        self.paddle_mount = None

    def forward(self, q):
        poses = {self.root: np.eye(4)}
        for parent, child, name, kind, origin, axis in self.joints:
            poses[child] = poses[parent]@origin
            if kind != 'fixed':
                poses[child] = poses[child]@transform(matrix=rotation(axis, float(q[self.index[name]])))
        to_pelvis = np.linalg.inv(poses['pelvis'])
        poses = {name: to_pelvis@pose for name, pose in poses.items()}
        if self.paddle_mount is not None:
            poses['h1_2_paddle_link'] = poses['right_wrist_yaw_link']@self.paddle_mount
        return poses


def protected_links(fk):
    # Shoulder pitch/roll housings are part of the joint attachment. Protect the
    # upper-arm yaw child, elbow, wrist links and paddle against torso AND pelvis.
    return [fk.child_by_joint[side+'_'+part+'_joint'] for side in ('left', 'right')
            for part in ARMS[2:]]+['h1_2_paddle_link']


def box_pose(local_bounds, pose):
    lo, hi = np.asarray(local_bounds)
    center = pose[:3, :3]@((lo+hi)*.5)+pose[:3, 3]
    return center, pose[:3, :3], (hi-lo)*.5


def separating_gap(a, b):
    """Largest separating-axis gap; positive is conservative box separation."""
    ca, ra, ea = a
    cb, rb, eb = b
    axes = np.concatenate((ra.T, rb.T, np.cross(ra.T[:, None, :], rb.T[None, :, :]).reshape(-1, 3)))
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-8]/norms[norms > 1e-8, None]
    gaps = np.abs(axes@(ca-cb))-np.abs(axes@ra)@ea-np.abs(axes@rb)@eb
    return float(gaps.max())


def front_clearance(paddle, torso, torso_bounds):
    center, orient, extent = paddle
    p = torso[:3, :3].T@(center-torso[:3, 3])
    rot = torso[:3, :3].T@orient
    return float(p[0]-np.abs(rot[0])@extent-np.asarray(torso_bounds)[1, 0])


class ArmGuard:
    def __init__(self, fk, bounds, limits, initial, dt):
        self.fk, self.bounds = fk, bounds
        self.limits = np.asarray(limits)
        self.dt = float(dt)
        self.right = [fk.index['right_'+part+'_joint'] for part in ARMS]
        self.arms = [fk.index[side+'_'+part+'_joint'] for side in ('left', 'right') for part in ARMS]
        self.watch = protected_links(fk)
        assert set(self.watch+['torso_link', 'pelvis']) <= bounds.keys()
        self.last = np.asarray(initial, dtype=float).copy()
        self.calls = self.restricted = 0
        self.active_calls = self.active_restricted = 0
        self.ready = None
        self.last_info = {}

    def inspect(self, q, require_front):
        poses = self.fk.forward(q)
        boxes = {name: box_pose(self.bounds[name], poses[name]) for name in self.bounds}
        gap, pair = math.inf, None
        for arm in self.watch:
            for body in ('torso_link', 'pelvis'):
                value = separating_gap(boxes[arm], boxes[body])
                if value < gap:
                    gap, pair = value, [arm, body]
        front = front_clearance(boxes['h1_2_paddle_link'], poses['torso_link'], self.bounds['torso_link'])
        wrist_local = np.linalg.inv(poses['torso_link'])@poses['right_wrist_yaw_link']
        wrist_front = float(wrist_local[0, 3]-self.bounds['torso_link'][1][0])
        info = dict(min_box_separation_axis_gap_m=gap, nearest_pair=pair,
                    paddle_front_clearance_m=front, wrist_front_clearance_m=wrist_front,
                    front_required=bool(require_front))
        return gap >= MARGIN_M and (not require_front or front >= FRONT_MARGIN_M and wrist_front >= .02), info

    def path_ok(self, actual, target, require_front):
        distance = float(np.max(np.abs(np.asarray(target)[self.arms]-np.asarray(actual)[self.arms])))
        count = max(2, math.ceil(distance/.025)+1)
        if count > 200:
            return False
        # Discrete joint-space sweep, not a formal continuous-time guarantee.
        return all(self.inspect(np.asarray(actual)*(1.-s)+np.asarray(target)*s, require_front)[0]
                   for s in np.linspace(0., 1., count))

    def prepare(self, initial, normal_local):
        """Validate a known attainable target against the LIVE FK/boxes/limits."""
        from ready_pose import (RIGHT_READY_RAD, REFERENCE_PADDLE_CENTER_TORSO_M,
                                REFERENCE_PADDLE_NORMAL_TORSO, SETUP_MOVE_S)
        initial = np.asarray(initial, dtype=float).copy()
        assert self.inspect(initial, False)[0], 'INITIAL CONSERVATIVE COLLISION CLEARANCE FAILED: '+str(self.inspect(initial, False)[1])
        ready = initial.copy()
        for name, value in RIGHT_READY_RAD.items():
            ready[self.fk.index[name]] = value
        assert np.isfinite(ready).all()
        assert np.all(ready >= self.limits[:, 0]) and np.all(ready <= self.limits[:, 1]), 'READY OUTSIDE LIVE JOINT LIMITS'
        assert np.min(np.minimum(ready[self.right]-self.limits[self.right, 0],
                                 self.limits[self.right, 1]-ready[self.right])) >= .10, 'READY LIMIT MARGIN TOO SMALL'
        poses = self.fk.forward(ready)
        pose = np.linalg.inv(poses['torso_link'])@poses['h1_2_paddle_link']
        normal = pose[:3, :3]@np.asarray(normal_local)
        position_error = float(np.linalg.norm(pose[:3, 3]-REFERENCE_PADDLE_CENTER_TORSO_M))
        normal_error = float(np.linalg.norm(normal-REFERENCE_PADDLE_NORMAL_TORSO))
        assert position_error < .002 and normal_error < .01, 'READY LIVE GEOMETRY DOES NOT MATCH VERIFIED CANDIDATE'
        ok, info = self.inspect(ready, True)
        assert ok, 'READY LIVE COLLISION/FRONT CHECK FAILED: '+str(info)
        assert self.path_ok(initial, ready, False), 'READY ENTRY PATH FAILED LIVE GEOMETRY CHECK'
        peak_speeds = np.abs(ready[self.arms]-initial[self.arms])*math.pi/(2.*SETUP_MOVE_S)
        caps = np.array([.9 if 'wrist' in self.fk.names[j] else .6 for j in self.arms])
        assert np.all(peak_speeds <= caps), 'READY SMOOTH PATH EXCEEDS ARM RATE LIMIT'
        # Validate the actual cosine-scheduled command sequence as well as the
        # .025-rad joint sweep. Both arms hold their initial values during entry.
        worst = math.inf
        for t in np.linspace(0., SETUP_MOVE_S, math.ceil(SETUP_MOVE_S/self.dt)+1):
            phase = .5*(1.-math.cos(math.pi*t/SETUP_MOVE_S))
            q = initial.copy()
            q[self.arms] += phase*(ready[self.arms]-initial[self.arms])
            clear, entry_info = self.inspect(q, False)
            assert clear, 'SCHEDULED READY ENTRY COLLISION: '+str(entry_info)
            worst = min(worst, entry_info['min_box_separation_axis_gap_m'])
        self.initial = initial.copy()
        self.ready = ready.copy()
        self.setup_move_s = SETUP_MOVE_S
        return dict(method='received_geometry_verified_candidate_cosine_entry',
                    ready_joint_targets=RIGHT_READY_RAD.copy(),
                    initial_joint_positions={n: float(initial[i]) for i, n in enumerate(self.fk.names)},
                    paddle_center_torso_m=pose[:3, 3].tolist(),
                    paddle_normal_torso=normal.tolist(),
                    position_error_m=position_error, normal_vector_error=normal_error,
                    pose_error=float(np.linalg.norm(np.r_[pose[:3, 3]-REFERENCE_PADDLE_CENTER_TORSO_M,
                                                         .2*(normal-REFERENCE_PADDLE_NORMAL_TORSO)])),
                    setup_move_s=SETUP_MOVE_S, settle_until_s=5.,
                    peak_requested_arm_speed_rad_s=peak_speeds.tolist(),
                    minimum_sampled_entry_gap_m=worst,
                    front_margin_m=FRONT_MARGIN_M, box_clearance_m=MARGIN_M,
                    ready_geometry=info)

    def select(self, actual, requested, time_s):
        actual, requested = np.asarray(actual), np.asarray(requested)
        self.calls += 1
        self.active_calls += int(time_s >= 5.5)
        require_front = time_s >= 5.
        if not self.inspect(actual, require_front)[0]:
            raise RuntimeError('ACTUAL POSE GUARD VIOLATION: '+str(self.inspect(actual, require_front)[1]))
        desired = requested.copy()
        if time_s < 5.:
            phase = .5*(1.-math.cos(math.pi*min(max(time_s, 0.), self.setup_move_s)/self.setup_move_s))
            desired[self.arms] = self.initial[self.arms]+phase*(self.ready[self.arms]-self.initial[self.arms])
        maximum = np.array([(.9 if 'wrist' in self.fk.names[j] else .6)*self.dt for j in self.arms])
        target = actual.copy()
        target[self.arms] = self.last[self.arms]+np.clip(desired[self.arms]-self.last[self.arms], -maximum, maximum)
        target[self.arms] = np.clip(target[self.arms], self.limits[self.arms, 0], self.limits[self.arms, 1])
        for fraction in (1., .5, .25, .125, 0.):
            candidate = actual.copy()
            candidate[self.arms] = self.last[self.arms]+fraction*(target[self.arms]-self.last[self.arms])
            if self.path_ok(actual, candidate, require_front):
                changed = fraction < 1. or np.max(np.abs(candidate[self.arms]-desired[self.arms])) > 1e-5
                self.restricted += int(changed)
                self.active_restricted += int(time_s >= 5.5 and changed)
                self.last = candidate.copy()
                self.last_info = dict(self.inspect(candidate, require_front)[1], accepted_fraction=fraction)
                result = requested.copy()
                result[self.arms] = candidate[self.arms]
                return result
        raise RuntimeError('NO COLLISION-CLEAR ARM TARGET PATH; stopped without joint teleport')
