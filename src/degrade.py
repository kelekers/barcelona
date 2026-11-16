"""Make clean, provider-corrected tracking look like automatic tracking.

The external games are corrected data: every player is present on every frame, positions are
accurate to centimetres and the ball is never lost. The competition ATD is none of those things.
Two kinds of transformation are applied to external games only.

1. Tracking corruptions -- LABEL-INDEPENDENT. Where the ball is dropped or a player is jittered
   is drawn at random, never from the label, so the model learns "with the ball missing, decide
   from the players". Each parameter mirrors a property measured on the UNLABELLED competition
   tracking (calibrate_degradation.py): the ball is absent in long blocks only (no gap of 26-75
   frames exists, so shorter occlusions arrive already interpolated), and about two of the 22
   players are missing on a typical frame.

2. Dead-ball ball replacement -- the one label-dependent step, and it REMOVES information.
   The DFL provider freezes the ball at a fixed point for the whole of every stoppage (its speed
   is exactly zero on over 90% of dead frames). No camera-based tracker behaves like that, and a
   model allowed to see it would learn "frozen ball = dead" and nothing else. So inside each
   stoppage the provider's ball is replaced by one of three things an automatic tracker actually
   produces: nothing at all, a ball that travels with a nearby player (carried back to the
   restart), or the stationary ball (placed for the set piece). The mix is an assumption, which
   is why the dead/alive head is not allowed to see the ball at all (model.py); the replacement
   only shapes what the team head sees in its windows next to a stoppage.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .common import D

SETTINGS = {
    "raw": None,
    "clean": dict(ball_sigma=0.0, ball_noise_m=0.03, player_noise_m=0.0, player_drop=0.0,
                  block_frac=0.10, interp_per_min=0.5, team_flip=0.0),
    "light": dict(ball_sigma=2.0, ball_noise_m=0.10, player_noise_m=0.35, player_drop=0.04,
                  block_frac=0.18, interp_per_min=1.0, team_flip=0.0),
    "heavy": dict(ball_sigma=3.5, ball_noise_m=0.20, player_noise_m=0.70, player_drop=0.09,
                  block_frac=0.30, interp_per_min=2.0, team_flip=0.01),
}

# Position error is slowly varying: the ATD's collective player speed distribution matches clean
# tracking, so the error must be a drift, not a flicker (flicker would inflate every speed).
PLAYER_NOISE_CORR = 60    # frames
BALL_NOISE_CORR = 10      # frames
BLOCK_MIN, BLOCK_MAX = 76, 2500     # frames; ATD has no ball gap between 26 and 75 frames
INTERP_MIN, INTERP_MAX = 8, 75      # shorter occlusions arrive already interpolated
DEAD_MODES = (("missing", 0.55), ("carried", 0.25), ("stationary", 0.20))


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _bridge(bx, by, a, b):
    """Linear interpolation over [a, b) between the neighbouring samples, if both exist."""
    if a < 1 or b >= len(bx) or not (np.isfinite(bx[a - 1]) and np.isfinite(bx[b])):
        bx[a:b], by[a:b] = np.nan, np.nan
        return
    t = np.linspace(0.0, 1.0, b - a + 2)[1:-1]
    bx[a:b] = bx[a - 1] + t * (bx[b] - bx[a - 1])
    by[a:b] = by[a - 1] + t * (by[b] - by[a - 1])


def randomize_dead_ball(match, rng):
    """Replace the provider's frozen dead-ball position (see module docstring, point 2)."""
    length, width = match["pitch"]
    bx = match["ball_x"].astype(np.float64).copy()
    by = match["ball_y"].astype(np.float64).copy()
    px, py = match["px"], match["py"]
    modes = [m for m, _ in DEAD_MODES]
    probs = [p for _, p in DEAD_MODES]
    for a, b in _runs(match["labels"] == D):
        mode = modes[int(rng.choice(len(modes), p=probs))]
        if mode == "stationary" or b - a < 6:
            continue
        # the ball is usually still tracked as it leaves play and again just before the restart
        lo = a + int(rng.integers(0, 40))
        hi = b - int(rng.integers(0, 30))
        if hi - lo < 4:
            continue
        if mode == "missing":
            if hi - lo < BLOCK_MIN:
                _bridge(bx, by, lo, hi)          # a short loss arrives interpolated
            else:
                bx[lo:hi], by[lo:hi] = np.nan, np.nan
        else:  # carried
            anchor = min(b - 1, lo + 5)
            dist = np.hypot((px[anchor] - bx[anchor]) * length, (py[anchor] - by[anchor]) * width)
            if not np.isfinite(dist).any():
                continue
            slot = int(np.nanargmin(dist))
            angle, radius = rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 1.2)
            bx[lo:hi] = px[lo:hi, slot] + radius * np.cos(angle) / length
            by[lo:hi] = py[lo:hi, slot] + radius * np.sin(angle) / width
    out = dict(match)
    out["ball_x"], out["ball_y"] = bx.astype(np.float32), by.astype(np.float32)
    return out


def degrade(match, setting, seed):
    """Corrupted copy of a match dictionary. `setting` is a key of SETTINGS."""
    params = SETTINGS[setting]
    if params is None:
        return match
    rng = np.random.default_rng(seed)
    if match.get("dead_ball_frozen") and match.get("labels") is not None:
        match = randomize_dead_ball(match, rng)
    length, width = match["pitch"]
    n = match["n_frames"]
    out = dict(match)
    px = match["px"].astype(np.float32).copy()
    py = match["py"].astype(np.float32).copy()
    bx = match["ball_x"].astype(np.float64).copy()
    by = match["ball_y"].astype(np.float64).copy()
    n_slots = px.shape[1]

    # Ball: slowly-varying measurement error, then the provider's temporal smoothing.
    for arr, scale in ((bx, length), (by, width)):
        noise = gaussian_filter1d(rng.normal(0.0, 1.0, n), BALL_NOISE_CORR, mode="nearest")
        arr += noise * (params["ball_noise_m"] / (noise.std() * scale))
    if params["ball_sigma"] > 0:
        for a, b in _runs(np.isfinite(bx)):
            if b - a >= 8:
                bx[a:b] = gaussian_filter1d(bx[a:b], params["ball_sigma"], mode="nearest")
                by[a:b] = gaussian_filter1d(by[a:b], params["ball_sigma"], mode="nearest")

    # Ball: short occlusions that the provider bridged with a straight line.
    for _ in range(int(params["interp_per_min"] * n / (25 * 60))):
        a = int(rng.integers(1, n - INTERP_MAX - 1))
        b = a + int(rng.integers(INTERP_MIN, INTERP_MAX))
        if np.isfinite(bx[a - 1]) and np.isfinite(bx[b]):
            _bridge(bx, by, a, b)

    # Ball: long blocks with no ball at all, placed at random.
    target, dropped = params["block_frac"] * n, 0
    while dropped < target:
        size = int(np.clip(rng.lognormal(mean=5.6, sigma=1.0), BLOCK_MIN, BLOCK_MAX))
        a = int(rng.integers(0, max(1, n - size)))
        bx[a:a + size], by[a:a + size] = np.nan, np.nan
        dropped += size

    # Players: slowly-varying position error (a tracker drifts, it does not flicker).
    if params["player_noise_m"] > 0:
        for arr, scale in ((px, length), (py, width)):
            noise = rng.normal(0.0, 1.0, arr.shape).astype(np.float32)
            noise = gaussian_filter1d(noise, PLAYER_NOISE_CORR, axis=0, mode="nearest")
            noise *= params["player_noise_m"] / (noise.std() * scale)
            arr += noise

    # Players: a slot goes missing for a while (occlusion, out of shot).
    if params["player_drop"] > 0:
        mean_burst = 60.0
        start_prob = params["player_drop"] / mean_burst
        for slot in range(n_slots):
            for a in np.flatnonzero(rng.random(n) < start_prob):
                b = a + int(rng.exponential(mean_burst)) + 5
                px[a:b, slot], py[a:b, slot] = np.nan, np.nan

    # Players: a tracklet occasionally carries the wrong team for a while. Implemented by moving
    # it into a spare slot that belongs to the other team.
    team_map = match["team_map"]
    if params["team_flip"] > 0:
        extra_x = np.full((n, 2), np.nan, np.float32)
        extra_y = np.full((n, 2), np.nan, np.float32)
        for slot in range(n_slots):
            other = 1 - int(team_map[slot])
            for a in np.flatnonzero(rng.random(n) < params["team_flip"] / 100.0):
                b = min(n, a + int(rng.exponential(100.0)) + 10)
                rows = np.arange(a, b)[~np.isfinite(extra_x[a:b, other])]
                extra_x[rows, other], extra_y[rows, other] = px[rows, slot], py[rows, slot]
                px[rows, slot], py[rows, slot] = np.nan, np.nan
        px = np.concatenate([px, extra_x], axis=1)
        py = np.concatenate([py, extra_y], axis=1)
        team_map = np.concatenate([team_map, np.array([0, 1], team_map.dtype)])

    out.update(px=px, py=py, ball_x=bx.astype(np.float32), ball_y=by.astype(np.float32),
               team_map=team_map)
    return out
