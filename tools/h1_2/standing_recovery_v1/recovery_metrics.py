"""Deterministic root-velocity test cases and first-episode recovery accounting."""
from collections import deque
import math

PUSH_TIME_S = 5.
BAND = {'height_tolerance_m': .05, 'tilt_rad': .08, 'xy_drift_m': .08,
        'linear_speed_mps': .10, 'angular_speed_rps': .30, 'foot_fz_N': 5.}
BASELINE_SECONDS = .5
HOLD_SECONDS = 1.
DEADLINE_SECONDS = 3.
TAIL_SECONDS = 3.
MIN_GOOD_FRACTION = .95
CASES = [
    {'name': f'{axis}_{sign}_{label}', 'delta_v_world_mps': [direction*speed if j == index else 0. for j in range(3)]}
    for speed, label in ((.05, '005'), (.10, '010'))
    for axis, index in (('x', 0), ('y', 1))
    for sign, direction in (('pos', 1.), ('neg', -1.))
]


def select_cases(name):
    return [dict(case, delta_v_world_mps=list(case['delta_v_world_mps'])) for case in CASES
            if name == 'all' or case['name'] == name]


class RecoveryMetrics:
    def __init__(self, episode_metrics, case):
        self.base = episode_metrics
        self.case = case
        self.baseline = deque(maxlen=max(1, round(BASELINE_SECONDS/self.base.dt)))
        self.tail = deque(maxlen=max(1, round(TAIL_SECONDS/self.base.dt)))
        self.push = None
        self.baseline_good_fraction = None
        self.baseline_settled = False
        self.good_streak = 0
        self.recovery_complete = None

    @property
    def ended(self):
        return self.base.ended

    @property
    def count(self):
        return self.base.count

    def in_band(self, row):
        return (abs(row['height_m']-self.base.reference) <= BAND['height_tolerance_m']
                and row['tilt_rad'] <= BAND['tilt_rad']
                and row['xy_drift_m'] <= BAND['xy_drift_m']
                and row['speed_mps'] <= BAND['linear_speed_mps']
                and row['angular_speed_rps'] <= BAND['angular_speed_rps']
                and len(row['foot_fz_N']) == 2 and all(f > BAND['foot_fz_N'] for f in row['foot_fz_N']))

    def mark_push(self, time_s, before, after):
        assert not self.ended and self.push is None, 'PUSH REPEATED OR APPLIED AFTER FIRST EPISODE'
        assert abs(time_s-PUSH_TIME_S) < self.base.dt/2, 'INCORRECT PUSH TIME'
        assert len(before) == len(after) == 6
        requested = self.case['delta_v_world_mps']+[0., 0., 0.]
        assert all(math.isfinite(v) for v in before+after)
        actual = [b-a for a, b in zip(before, after)]
        assert all(abs(x-y) < 1e-5 for x, y in zip(actual, requested)), 'VELOCITY WRITE DID NOT MATCH REQUEST'
        self.baseline_good_fraction = sum(self.baseline)/len(self.baseline) if self.baseline else None
        self.baseline_settled = (len(self.baseline) == self.baseline.maxlen
                                 and self.baseline_good_fraction >= MIN_GOOD_FRACTION)
        self.push = {'time_s': time_s, 'velocity_before_world': before,
                     'velocity_after_world': after, 'requested_delta_v_world_mps': requested,
                     'actual_delta_v_world_mps': actual, 'baseline_settled': self.baseline_settled}

    def add(self, row):
        if self.ended:
            return
        assert math.isfinite(row['angular_speed_rps'])
        self.base.add(row)
        good = self.in_band(row)
        if self.push is None:
            self.baseline.append(good)
        else:
            self.tail.append(good)
            self.good_streak = self.good_streak+1 if good else 0
            if self.recovery_complete is None and self.good_streak >= round(HOLD_SECONDS/self.base.dt):
                self.recovery_complete = row['time_s']

    def result(self):
        base = self.base.result()
        latency = self.recovery_complete-self.push['time_s'] if self.recovery_complete is not None else None
        recovered = latency is not None and latency <= DEADLINE_SECONDS+self.base.dt/2
        tail_fraction = sum(self.tail)/len(self.tail) if self.tail else None
        tail_good = len(self.tail) == self.tail.maxlen and tail_fraction >= MIN_GOOD_FRACTION
        passed = (self.push is not None and self.baseline_settled and recovered and tail_good
                  and base['stage1_candidate_pass'])
        reason = ('push_not_reached' if self.push is None else
                  'pre_push_not_settled' if not self.baseline_settled else
                  'standing_acceptance_failed' if not base['stage1_candidate_pass'] else
                  'recovery_deadline_missed' if not recovered else
                  'final_window_not_settled' if not tail_good else 'passed')
        return {'case': self.case, 'push': self.push, 'baseline_good_fraction': self.baseline_good_fraction,
                'recovery_window_complete_s': self.recovery_complete, 'recovery_latency_s': latency,
                'recovered_by_deadline': recovered, 'final_window_good_fraction': tail_fraction,
                'recovery_pass': passed, 'reason': reason, 'standing': base}
