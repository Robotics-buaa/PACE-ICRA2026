"""First-episode statistics; never stitch auto-reset episodes into a 20s success."""
import math


THRESHOLDS = {
    'height_tolerance_m': .08, 'max_tilt_rad': .15, 'max_xy_drift_m': .10,
    'min_both_feet_contact_fraction': .90,
    'estimated_load_near_limit_ratio': .95, 'max_near_limit_fraction': .05,
    'max_estimated_load_peak_ratio': 1.2,
}


class EpisodeMetrics:
    def __init__(self, joint_names, reference_height, dt, seconds=20.):
        self.names = list(joint_names)
        self.reference = reference_height
        self.dt, self.seconds = dt, seconds
        self.count = 0
        self.ended = False
        self.duration = 0.
        self.timeout = False
        self.physical_failure = False
        self.minimum_height, self.maximum_height = math.inf, -math.inf
        self.max_tilt = self.max_drift = self.max_speed = 0.
        self.both_contact_count = 0
        self.error_squared_sum = [0.]*len(self.names)
        self.error_peak = [0.]*len(self.names)
        self.load_peak = [0.]*len(self.names)
        self.near_count = [0]*len(self.names)
        self.projected_request_steps = 0

    def add(self, row):
        if self.ended:
            return  # states after the first automatic reset never affect acceptance
        values = [row[k] for k in ('time_s', 'height_m', 'tilt_rad', 'xy_drift_m', 'speed_mps')]
        values += row['foot_fz_N']+row['tracking_error_rad']+row['estimated_load_ratio']
        assert all(math.isfinite(v) for v in values), 'NONFINITE EVALUATION METRIC'
        assert len(row['tracking_error_rad']) == len(row['estimated_load_ratio']) == len(self.names)
        self.count += 1
        self.duration = row['time_s']
        self.minimum_height = min(self.minimum_height, row['height_m'])
        self.maximum_height = max(self.maximum_height, row['height_m'])
        self.max_tilt = max(self.max_tilt, row['tilt_rad'])
        self.max_drift = max(self.max_drift, row['xy_drift_m'])
        self.max_speed = max(self.max_speed, row['speed_mps'])
        self.both_contact_count += int(len(row['foot_fz_N']) == 2 and all(f > 5. for f in row['foot_fz_N']))
        self.projected_request_steps += int(row['projection_applied'])
        for j, (error, load) in enumerate(zip(row['tracking_error_rad'], row['estimated_load_ratio'])):
            self.error_squared_sum[j] += error*error
            self.error_peak[j] = max(self.error_peak[j], abs(error))
            self.load_peak[j] = max(self.load_peak[j], load)
            self.near_count[j] += int(load >= THRESHOLDS['estimated_load_near_limit_ratio'])
        # Same physical boundaries as the unchanged standing task, including a
        # simultaneous failure+timeout that reset & ~timeout alone would hide.
        physical = (row['height_m'] < self.reference-.18 or row['tilt_rad'] > .45
                    or row['xy_drift_m'] > .5)
        self.physical_failure |= physical or (row['reset'] and not row['timeout'])
        if row['reset'] or self.duration >= self.seconds-self.dt/2:
            self.ended = True
            self.timeout = bool(row['timeout'])

    def result(self):
        assert self.count, 'NO FIRST-EPISODE DATA'
        complete = (self.duration >= self.seconds-self.dt/2
                    and self.count*self.dt >= self.seconds-self.dt/2
                    and not self.physical_failure)
        support = self.both_contact_count/self.count
        kinematic = (complete and self.minimum_height >= self.reference-THRESHOLDS['height_tolerance_m']
                     and self.maximum_height <= self.reference+THRESHOLDS['height_tolerance_m']
                     and self.max_tilt <= THRESHOLDS['max_tilt_rad']
                     and self.max_drift <= THRESHOLDS['max_xy_drift_m']
                     and support >= THRESHOLDS['min_both_feet_contact_fraction'])
        per_joint = {name: {
            'tracking_rms_rad': math.sqrt(self.error_squared_sum[j]/self.count),
            'tracking_peak_rad': self.error_peak[j], 'estimated_load_peak_ratio': self.load_peak[j],
            'estimated_near_limit_fraction': self.near_count[j]/self.count,
        } for j, name in enumerate(self.names)}
        load_flags = [name for name, item in per_joint.items()
                      if item['estimated_near_limit_fraction'] > THRESHOLDS['max_near_limit_fraction']
                      or item['estimated_load_peak_ratio'] > THRESHOLDS['max_estimated_load_peak_ratio']]
        return {'duration_s': self.duration, 'samples': self.count, 'timeout': self.timeout,
                'physical_failure': self.physical_failure, 'completed_20s': complete,
                'min_height_m': self.minimum_height, 'max_height_m': self.maximum_height,
                'max_tilt_rad': self.max_tilt, 'max_xy_drift_m': self.max_drift,
                'max_speed_mps': self.max_speed, 'both_feet_contact_fraction': support,
                'projected_request_steps': self.projected_request_steps,
                'kinematic_standing_pass': kinematic, 'estimated_load_flag_joints': load_flags,
                'stage1_candidate_pass': kinematic and not load_flags, 'per_joint': per_joint}
