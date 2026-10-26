"""Separate standing/TT observation semantics and bounded inference composition."""
import math

ARM_PARTS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
             'wrist_roll', 'wrist_pitch', 'wrist_yaw')


def right_arm_indices(names):
    expected = {'right_'+part+'_joint' for part in ARM_PARTS}
    result = [i for i, name in enumerate(names) if name in expected]
    assert len(names) == len(set(names)) == 27 and len(result) == 7
    return result


def arm_blend(time_s):
    # First5s: right arm default target. Next.5s: smooth transition to TT commands.
    phase = min(1., max(0., (time_s-5.)/.5))
    return .5-.5*math.cos(math.pi*phase)


def standing_view(tt_obs, paddle_goal, reference_offset, goal_xy, scales, xp):
    """Convert each 99-field TT history frame, preserving its own time/heading.

    TT rel=(prediction_xy-R_heading*offset_xy-table_offset-robot_table_xy)*scale.
    Stand rel=(goal_xy-table_offset-robot_table_xy)*scale. Their difference
    cancels the table offset, so old history frames require no current pose alias.
    Noise/delay must be disabled; check latest transformed frame against actual pose.
    """
    ball_scale, robot_scale, gravity_scale = scales
    assert all(math.isfinite(v) and v != 0 for v in scales)
    assert tt_obs.shape[-1] == 495
    source = tt_obs.reshape(-1, 5, 99)
    out = source.clone() if hasattr(source, 'clone') else source.copy()
    prediction_xy = source[:, :, 93:95]/ball_scale
    heading = source[:, :, 98]/gravity_scale
    ox, oy = reference_offset[:, 0:1], reference_offset[:, 1:2]
    rotated_x = xp.cos(heading)*ox-xp.sin(heading)*oy
    rotated_y = xp.sin(heading)*ox+xp.cos(heading)*oy
    out[:, :, 96] += (goal_xy[0]-prediction_xy[:, :, 0]+rotated_x)*robot_scale
    out[:, :, 97] += (goal_xy[1]-prediction_xy[:, :, 1]+rotated_y)*robot_scale
    out[:, :, 87:93] = 0.
    out[:, :, 93:96] = paddle_goal[:, None, :]*ball_scale
    return out.reshape(-1, 495)


def contact_windows(samples, contacts, dt):
    """Pair-force evidence, plus complete2s body-support windows after contact.

    A contact is a measured filtered normal-force observation, not a return success.
    Sparse/short post-contact windows remain unverified. No reset episodes stitched.
    """
    events = []
    for hit in contacts:
        if hit['time_s'] < 5.5 or hit.get('arm_blend', 1.) < 1.:  # full TT arm authority only
            continue
        if events and hit['time_s']-events[-1]['last_contact_s'] <= .1+1e-8:
            events[-1]['last_contact_s'] = hit['time_s']
            events[-1]['peak_pair_force_N'] = max(events[-1]['peak_pair_force_N'], hit['force_N'])
        else:
            events.append({'first_contact_s': hit['time_s'], 'last_contact_s': hit['time_s'],
                           'peak_pair_force_N': hit['force_N']})
    for event in events:
        t = event['first_contact_s']
        start = math.ceil((t-1e-8)/dt)*dt
        end = start+2.
        first_step, last_step = round(start/dt), round(end/dt)
        indexed = [(round(r['time_s']/dt), r) for r in samples]
        assert all(abs(step*dt-r['time_s']) < dt*.01 for step, r in indexed), 'OFF-GRID CONTROL TIMESTAMP'
        selected = [(step, r) for step, r in indexed if first_step <= step <= last_step]
        rows = [r for step, r in selected]
        complete = [step for step, r in selected] == list(range(first_step, last_step+1))
        support = sum(all(f > 5. for f in r['foot_fz_N']) for r in rows)/len(rows) if rows else None
        good = bool(complete and support >= .9 and all(
            abs(r['height_m']-r['reference_height_m']) <= .08
            and r['tilt_rad'] <= .15 and r['xy_drift_m'] <= .10
            and not r['physical_failure'] and max(r['estimated_load_ratio']) <= 1.2
            for r in rows))
        event.update(post_window_start_s=start, post_window_end_s=end,
                     post_window_complete=complete, post_window_both_feet_fraction=support,
                     post_window_body_support_pass=good)
    return events
