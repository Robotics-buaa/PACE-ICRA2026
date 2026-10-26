"""Same quiet band as recovery; statistics at every control step, no new thresholds."""
import math

CONDITIONS = ('right_arm_default', 'both_arms_default')


def arm_action_indices(names, condition):
    assert condition in CONDITIONS
    sides = ('right_',) if condition == 'right_arm_default' else ('left_', 'right_')
    result = [i for i, name in enumerate(names) if name.startswith(sides)
              and any(part in name for part in ('_shoulder_', '_elbow_', '_wrist_'))]
    assert len(result) == (7 if condition == 'right_arm_default' else 14), 'ARM ACTION MAPPING CHANGED'
    return result


def misses(row, reference, band):
    return {
        'height': abs(row['height_m']-reference) > band['height_tolerance_m'],
        'tilt': row['tilt_rad'] > band['tilt_rad'],
        'drift': row['xy_drift_m'] > band['xy_drift_m'],
        'linear_speed': row['speed_mps'] > band['linear_speed_mps'],
        'angular_speed': row['angular_speed_rps'] > band['angular_speed_rps'],
        'foot_contact': len(row['foot_fz_N']) != 2 or any(f <= band['foot_fz_N'] for f in row['foot_fz_N']),
    }


class QuietMetrics:
    def __init__(self, base, condition, band):
        self.base, self.condition, self.band = base, condition, dict(band)
        self.windows = {name: {'samples': 0, 'good': 0, 'miss_counts': {key: 0 for key in
                        ('height', 'tilt', 'drift', 'linear_speed', 'angular_speed', 'foot_contact')},
                        'max_speed_mps': 0., 'max_angular_speed_rps': 0.,
                        'joint_velocity_sqsum': [0.]*len(base.names)} for name in ('all', 'pre5s', 'tail')}
        self.previous_targets = None
        self.target_step_sqsum = [0.]*len(base.names)
        self.target_step_peak = [0.]*len(base.names)

    @property
    def ended(self):
        return self.base.ended

    @property
    def count(self):
        return self.base.count

    def add(self, row):
        if self.ended:
            return
        self.base.add(row)
        assert math.isfinite(row['angular_speed_rps'])
        assert len(row['joint_velocity_radps']) == len(row['joint_target_rad']) == len(self.base.names)
        assert all(math.isfinite(v) for v in row['joint_velocity_radps']+row['joint_target_rad'])
        failed = misses(row, self.base.reference, self.band)
        for name, window in self.windows.items():
            t = row['time_s']
            include = name == 'all' or name == 'pre5s' and 4.5+1e-6 < t <= 5.+1e-6 or name == 'tail' and 17.+1e-6 < t <= 20.+1e-6
            if not include:
                continue
            window['samples'] += 1
            window['good'] += int(not any(failed.values()))
            window['max_speed_mps'] = max(window['max_speed_mps'], row['speed_mps'])
            window['max_angular_speed_rps'] = max(window['max_angular_speed_rps'], row['angular_speed_rps'])
            for key, value in failed.items():
                window['miss_counts'][key] += int(value)
            for j, velocity in enumerate(row['joint_velocity_radps']):
                window['joint_velocity_sqsum'][j] += velocity*velocity
        if self.previous_targets is not None:
            for j, (old, current) in enumerate(zip(self.previous_targets, row['joint_target_rad'])):
                step = current-old
                self.target_step_sqsum[j] += step*step
                self.target_step_peak[j] = max(self.target_step_peak[j], abs(step))
        self.previous_targets = list(row['joint_target_rad'])

    def result(self):
        base = self.base.result()
        windows = {}
        for name, window in self.windows.items():
            count = window['samples']
            windows[name] = {key: value for key, value in window.items() if key != 'joint_velocity_sqsum'}
            windows[name]['good_fraction'] = window['good']/count if count else None
            windows[name]['joint_velocity_rms_radps'] = {
                joint: math.sqrt(window['joint_velocity_sqsum'][j]/count) if count else None
                for j, joint in enumerate(self.base.names)}
        pre, tail = windows['pre5s'], windows['tail']
        quiet_pass = (base['stage1_candidate_pass'] and pre['samples'] == round(.5/self.base.dt)
                      and tail['samples'] == round(3./self.base.dt)
                      and pre['good_fraction'] >= .95 and tail['good_fraction'] >= .95)
        return {'condition': self.condition, 'standing': base, 'windows': windows,
                'quiet_candidate_pass': quiet_pass,
                'joint_target_step_rms_rad': {joint: math.sqrt(self.target_step_sqsum[j]/max(1, self.base.count-1))
                                              for j, joint in enumerate(self.base.names)},
                'joint_target_step_peak_rad': dict(zip(self.base.names, self.target_step_peak))}
