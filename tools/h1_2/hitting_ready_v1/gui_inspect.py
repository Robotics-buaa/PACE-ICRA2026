"""Render a paused diagnostic scene without advancing simulated time."""
import time


def paused_view(app, sim, seconds=None):
    sim.pause()
    start = time.perf_counter()
    while app.is_running():
        # Reassert pause if a UI action changed the timeline. Never call step().
        sim.pause()
        sim.render()
        if seconds is not None and time.perf_counter()-start >= seconds:
            return True
        time.sleep(1./60.)
    return False


def initial_preview(app, sim, seconds=5.):
    print('[READY-GUI] Initial READY pose: physics PAUSED for 5s. Static preview is not a hold acceptance.', flush=True)
    if not paused_view(app, sim, seconds):
        raise RuntimeError('GUI CLOSED DURING INITIAL STATIC PREVIEW')
    sim.play()
    print('[READY-GUI] Starting the NEW physical zero-action diagnostic.', flush=True)


def inspect_until_closed(app, sim, completed):
    status = 'Diagnostic completed' if completed else 'Diagnostic stopped/failed'
    print('[READY-GUI]', status+'. Physics PAUSED; terminal pose retained. Orbit the camera; CLOSE the window to exit. No training.', flush=True)
    paused_view(app, sim)
