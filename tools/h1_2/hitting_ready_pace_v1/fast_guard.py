"""Coalesce original 128-env sampling grids, with unchanged guard decisions.

One host transfer of block maxima replaces one synchronization per old block.
Every original candidate and sample remains evaluated, including rejected ones.
Temporary sampled FK/SAT batches are bounded independently of environment count.
"""
import math
from target_guard import SPACING_RAD, MAX_INTERVALS


def choose_batched(q, velocity, previous, proposed, arms, checked, rate, dt,
                   geom, inner, lookahead_s, xp, array, hard, ready,
                   block_size=128, sample_budget=10240):
    n = len(q)
    assert n > 0 and sample_budget >= 5*(MAX_INTERVALS+1)
    forecast = geom.inspect(q+velocity*(dt+lookahead_s))
    recovery = xp.any(forecast < inner, -1)
    forecast_ok = xp.all(forecast >= hard, -1)
    recover_target = proposed.clone() if hasattr(proposed, 'clone') else proposed.copy()
    delta = ready[:, arms]-previous[:, arms]
    recover_target[:, arms] = previous[:, arms]+xp.maximum(xp.minimum(delta, rate*dt), -rate*dt)
    effective = xp.where(recovery[:, None], recover_target, proposed)
    fractions = array([1., .5, .25, .125, 0.])[None, :, None]
    candidate = effective[:, None, :] + fractions*0.
    candidate = candidate.clone() if hasattr(candidate, 'clone') else candidate.copy()
    candidate[:, :, arms] = previous[:, None, arms]+fractions*(effective[:, None, arms]-previous[:, None, arms])
    distance = xp.amax(xp.abs(candidate[:, :, checked]-q[:, None, checked]), -1)
    rate_ok = xp.all(xp.abs(candidate[:, :, arms]-previous[:, None, arms]) <= rate[None, None, :]*dt+1e-6, -1)
    # Preserve each OLD block's exact maximum, interval count and alpha grid.
    # Group reduction is on GPU; only its small result is copied to the host.
    maxima = xp.stack([xp.amax(distance[s:s+block_size]) for s in range(0,n,block_size)])
    values = maxima.detach().cpu().tolist() if xp.__name__ == 'torch' else maxima.tolist()
    assert all(math.isfinite(v) for v in values), 'NONFINITE GUARD TARGET DISTANCE'
    groups = {}
    for b, maximum in enumerate(values):
        intervals = min(MAX_INTERVALS,max(1,math.ceil(maximum/SPACING_RAD)))
        groups.setdefault(intervals,[]).extend(range(b*block_size,min((b+1)*block_size,n)))
    path_minimum = distance[...,None]*array([0.,0.,0.])
    actual_before = forecast*0.
    count = distance[:,0]*0.
    density_ok = rate_ok & False
    for intervals, ids in groups.items():
        alpha = array([i/intervals for i in range(intervals+1)])[None,None,:,None]
        capacity = max(block_size,sample_budget//(5*(intervals+1)))
        for start in range(0,len(ids),capacity):
            rows = ids[start:start+capacity]
            sweep = q[rows,None,None,:]*(1.-alpha)+candidate[rows,:,None,:]*alpha
            clear = geom.inspect(sweep.reshape(-1,q.shape[-1])).reshape(len(rows),5,intervals+1,3)
            path_minimum[rows] = xp.amin(clear,-2)
            actual_before[rows] = clear[:,0,0,:]
            count[rows] = intervals+1
            density_ok[rows] = distance[rows] <= intervals*SPACING_RAD+1e-7
    path_ok = xp.all(path_minimum >= inner,-1)
    safe = path_ok & rate_ok & density_ok & forecast_ok[:,None]
    found = xp.any(safe,-1)
    if xp.__name__ == 'torch':
        first = safe.to(dtype=q.dtype).argmax(-1)
        rows = xp.arange(n,device=q.device)
    else:
        first = safe.astype(float).argmax(-1)
        rows = xp.arange(n)
    selected = xp.where(found[:,None],candidate[rows,first],q)
    return selected,found,dict(
        guard_pre_q=q,guard_pre_velocity=velocity,
        guard_previous_target=previous,guard_policy_proposed_target=proposed,
        guard_proposed_target=effective,guard_recovery_active=recovery,
        guard_candidate_min_margins=path_minimum,guard_candidate_distance=distance,
        guard_candidate_rate_ok=rate_ok,guard_candidate_path_ok=path_ok,
        guard_candidate_density_ok=density_ok,guard_forecast_margins=forecast,
        guard_forecast_ok=forecast_ok,
        guard_selected_fraction=xp.where(found,fractions[0,first,0],array(-1.)),
        guard_sample_count=count,guard_actual_before_margins=actual_before)
