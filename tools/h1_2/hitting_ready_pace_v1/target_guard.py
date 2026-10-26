"""Adaptive sampled arm-target guard, with numpy/torch backends for comparison.

Rate limits apply to changes of PD targets, not actual-target tracking error.
The sweep keeps a <=.025rad joint spacing. A capacity excess is rejected rather
than silently coarsening the sweep. Actual/candidate inner margins and hard
forecast limits are explicit below; soft forecast margins trigger recovery.
Sampling remains a screen, not a continuous-time guarantee.
"""
import math

SPACING_RAD = .025
MAX_INTERVALS = 64


def choose_targets(q, velocity, previous, proposed, arms, checked, rate, dt,
                   geom, inner, lookahead_s, xp, array, hard, ready):
    n = q.shape[0]
    forecast = geom.inspect(q+velocity*(dt+lookahead_s))
    recovery = xp.any(forecast < inner, -1)
    forecast_ok = xp.all(forecast >= hard, -1)
    # Inner forecast margins are a braking/recovery trigger; they must not
    # indiscriminately reject all controls when there is still hard clearance.
    # Recover toward the already validated ready target, at the same target-rate
    # caps. Actual/candidate paths retain INNER clearance; forecast retains HARD
    # clearance. No candidate is allowed to bypass either screen.
    recover_target = proposed.clone() if hasattr(proposed, 'clone') else proposed.copy()
    delta = ready[:, arms]-previous[:, arms]
    delta = xp.maximum(xp.minimum(delta, rate*dt), -rate*dt)
    recover_target[:, arms] = previous[:, arms]+delta
    effective = xp.where(recovery[:, None], recover_target, proposed)
    fractions = array([1., .5, .25, .125, 0.])[None, :, None]
    # Non-arm standing channels remain exactly the projected proposal.
    candidate = effective[:, None, :] + fractions*0.
    candidate = candidate.clone() if hasattr(candidate, 'clone') else candidate.copy()
    candidate[:, :, arms] = (previous[:, None, arms]
                            + fractions*(effective[:, None, arms]-previous[:, None, arms]))
    difference = xp.abs(candidate[:, :, checked]-q[:, None, checked])
    distance = xp.amax(difference, -1)
    rate_ok = xp.all(xp.abs(candidate[:, :, arms]-previous[:, None, arms])
                     <= rate[None, None, :]*dt+1e-6, -1)
    maximum = float(xp.amax(distance).item())
    assert math.isfinite(maximum), 'NONFINITE GUARD TARGET DISTANCE'
    intervals = min(MAX_INTERVALS, max(1, math.ceil(maximum/SPACING_RAD)))
    alpha = array([i/intervals for i in range(intervals+1)])[None, None, :, None]
    sweep = q[:, None, None, :]*(1.-alpha)+candidate[:, :, None, :]*alpha
    clear = geom.inspect(sweep.reshape(-1, q.shape[-1])).reshape(n, 5, intervals+1, 3)
    density_ok = distance <= intervals*SPACING_RAD+1e-7
    path_minimum = xp.amin(clear, -2)
    path_ok = xp.all(path_minimum >= inner, -1)
    safe = path_ok & rate_ok & density_ok & forecast_ok[:, None]
    found = xp.any(safe, -1)
    if xp.__name__ == 'torch':
        first = safe.to(dtype=q.dtype).argmax(-1)
        rows = xp.arange(n, device=q.device)
    else:
        first = safe.astype(float).argmax(-1)
        rows = xp.arange(n)
    chosen = candidate[rows, first]
    selected = xp.where(found[:, None], chosen, q)
    selected_fraction = xp.where(found, fractions[0, first, 0], array(-1.))
    diagnostic = dict(
        guard_pre_q=q, guard_pre_velocity=velocity,
        guard_previous_target=previous, guard_policy_proposed_target=proposed,
        guard_proposed_target=effective, guard_recovery_active=recovery,
        guard_candidate_min_margins=path_minimum,
        guard_candidate_distance=distance, guard_candidate_rate_ok=rate_ok,
        guard_candidate_path_ok=path_ok, guard_candidate_density_ok=density_ok,
        guard_forecast_margins=forecast, guard_forecast_ok=forecast_ok,
        guard_selected_fraction=selected_fraction,
        guard_sample_count=distance[:, 0]*0.+intervals+1,
        guard_actual_before_margins=clear[:, 0, 0, :])
    return selected, found, diagnostic
