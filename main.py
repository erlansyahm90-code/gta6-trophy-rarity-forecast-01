"""Forecast GTA VI trophy earn rates on Exophase at a case's target time.

Model: earner-count decomposition with a log-log trend, dilution, and a
refresh-aware spread.  Written for make.fun competition
gta6-trophy-rarity-forecast-01 (Python 3.13, standard library only).

Why this shape
--------------
An earn rate is a ratio:  rate = earners / players_tracked.  Over launch week
both parts move fast, and they move for different reasons:

* players_tracked grows as more players link their PSN to Exophase.  Every
  new tracked player starts with (nearly) no trophies, so a jump in tracked
  players DILUTES the rate of every trophy the newcomers have not earned yet.
  The house saw exactly this on launch night: one refresh took tracked
  players 638 -> 3,868 and every rate but the 100% one fell.
* earners grow as the tracked base plays: story trophies fast (players reach
  each chapter beat in order), grind / collectible / platinum trophies slowly.

Fitting the rate directly confuses the two.  So this program fits each side
separately in log space against log time since release, then divides:

    E_i(t) ~ exp(a_i + b_i * ln(1 + t))     earners of trophy i
    P(t)   ~ exp(a_P + b_P * ln(1 + t))     players tracked
    r_i    = 100 * E_i(target) / P(target)

Details that matter:

* Exophase refreshes on its own schedule, so the house's reads repeat the
  same page for hours.  Consecutive identical reads are collapsed to one
  snapshot before fitting; otherwise the fit is dominated by frozen
  plateaus and the last-24h window holds only a few distinct states with
  wildly unequal weight.
* Fits are weighted least squares over the last 24h of distinct snapshots,
  with recency weights (tau = 24h) and a sqrt(P) precision weight, so the
  launch-night freeze and the first refresh do not drag the slope.  A 24h
  window beats 48h in backtest: launch growth decelerates, so the recent
  slope is the one that extrapolates.
* Each trophy's earner slope is shrunk toward its tier's mean slope
  (empirical Bayes, pseudo-count 6): trophies with few distinct reads borrow
  strength from their tier instead of extrapolating noise.
* Extrapolated slopes are damped to 80% around the fit's centre: growth
  keeps decelerating past the fit window, so the undamped line overshoots
  the 4.5-day target (backtest: damping 0.8 clearly beats 1.0 and 0.6).
* The answering read shows the page as of Exophase's last refresh before
  it, so the forecast is evaluated at target minus half the typical
  refresh interval, estimated from the gaps between distinct snapshots.
* The spread is honest about four things: fit residual x leverage, the
  players-tracked fit's residual, refresh-lag uncertainty, and a
  random-walk term that grows with the horizon.  It is floored and capped.
* When the history is a single frozen page (the practice cases), there is
  no trend to fit.  The fallback then applies the launch-night dilution
  prior: the next refresh multiplies tracked players several-fold while
  earners grow only a little, so rates fall, with a wide spread.
* A trophy at 100% stays (near) 100%: every tracked player earns the
  prologue trophy within minutes, and it survived the 6x refresh at 100%.

Safety: any per-trophy failure falls back to persistence plus a random-walk
spread (the house baseline's shape), and any whole-case failure falls back
the same way for every trophy, so the program always writes a valid file.

Determinism: pure float arithmetic, no randomness, files processed in name
order, so two runs write byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path; this file needs no sibling module,
# but keep the directory first anyway so a future helper module would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Game release, the time origin for the log-time fits (game released 19 Nov
# 2026; the house's curve baseline uses the same origin).
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# --- model constants (tuned on the synthetic launch-week backtest) ---
SIGMA_RW = 0.25          # logit random-walk vol per sqrt-day.  The backtest
                         # prefers 0.20, but the simulation's noise is a
                         # lower bound on the real world's (weekend surge,
                         # Exophase quirks, model error), and pinball
                         # punishes an over-narrow spread more than a
                         # slightly wide one, so 0.25 buys insurance for a
                         # ~7% in-simulation cost.
SLOPE_DAMP = 0.8         # extrapolation damping for log-log slopes: launch
                         # growth decelerates, so a straight log-log line
                         # overshoots the target; 0.8 pivots the
                         # extrapolation toward flat around the fit's
                         # centre (backtest optimum is 0.75-0.8; 0.6 is
                         # clearly too much, 1.0 clearly too little)
SIGMA_FLOOR = 0.20       # never claim to know a rate better than this
SIGMA_CAP = 1.50         # never spread wider than this
FIT_WINDOW_HOURS = 24.0  # fit on distinct snapshots this recent (or all
                         # if fewer than 3 fall in the window): growth
                         # decelerates, so the recent slope is the one that
                         # extrapolates; 24h clearly beats 48h in backtest
RECENCY_TAU_HOURS = 24.0  # recency weight decay for the fits
SHRINK_PRIOR_N = 6.0     # pseudo-count shrinking a slope to its tier mean
E_GROWTH_CAP = 15.0      # earners may at most 15x over the horizon: a
                         # mid-story trophy's rate can ~4x while tracked
                         # players ~3x, so earners can ~12x; 15 caps the
                         # extrapolation without binding on real moves
E_GROWTH_FLOOR = 0.8     # earners may fall at most 20% (unlinking is rare)
P_GROWTH_CAP = 6.0       # tracked players may at most 6x over the horizon
LOGIT_MOVE_CAP = 3.0     # median may move at most ~3 logit units from last
DEFAULT_RESID = 0.25     # log-earner residual std when nothing is estimable
FLAT_SIGMA_LAUNCH = 1.0  # spread for a frozen launch-night page
FLAT_SIGMA_LATE = 0.7    # spread for a frozen page later in the week
LAUNCH_WINDOW_HOURS = 36.0  # a frozen page this soon after release is launch


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp (with or without fractions / Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    return (moment - RELEASE).total_seconds() / 3600.0


def log_time(t_hours):
    """Log-time regressor; floored so pre-release reads stay finite."""
    return math.log(max(1.0 + t_hours, 0.05))


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation, ~1e-9)."""
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if p <= p_high:
        q = p - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
    den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    return -num / den


def wls(xs, ys, ws):
    """Weighted least squares of y on x.  Returns a dict of fit statistics.

    intercept, slope: the fitted line.  resid: weighted residual std (0 when
    the fit is exact, e.g. two points).  n: number of points.  xbar, sxx:
    weighted mean of x and weighted sum of squared x deviations, for the
    leverage of a prediction at a new x.
    """
    sw = sum(ws)
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return {"intercept": ybar, "slope": 0.0, "resid": 0.0, "n": len(xs),
                "xbar": xbar, "sxx": 0.0, "sw": sw}
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    dof = max(len(xs) - 2, 1)
    resid_sq = sum(w * (y - (intercept + slope * x)) ** 2
                   for x, y, w in zip(xs, ys, ws)) / sw
    resid = math.sqrt(resid_sq * len(xs) / dof) if len(xs) > 2 else 0.0
    return {"intercept": intercept, "slope": slope, "resid": resid,
            "n": len(xs), "xbar": xbar, "sxx": sxx, "sw": sw}


def leverage_multiplier(fit, x_star):
    """sqrt(1 + prediction variance factor) at x_star, capped at 3."""
    if fit["sxx"] <= 1e-12 or fit["n"] == 0:
        return 1.0
    lev = 1.0 / fit["n"] + (x_star - fit["xbar"]) ** 2 / fit["sxx"]
    return min(math.sqrt(1.0 + lev), 3.0)


def extrapolate(fit, x_star, damp, slope=None):
    """Fitted value at x_star, with the slope damped around the fit centre.

    With damp = 1 and the fit's own slope this is the plain fitted line.
    With damp < 1 the line pivots toward flat around xbar, so the level at
    the data's centre is preserved while the far extrapolation is pulled
    back -- the honest shape for growth that decelerates.  `slope` lets a
    caller extrapolate with a shrunk slope while keeping the fit's level.
    """
    centre = fit["intercept"] + fit["slope"] * fit["xbar"]
    s = fit["slope"] if slope is None else slope
    return centre + s * damp * (x_star - fit["xbar"])


def is_story_trophy(trophy):
    """Cheap description classifier used only by the frozen-page fallback:
    chapter beats and mission completions, not counted grinds."""
    desc = trophy.get("description", "")
    if desc.startswith("Complete Chapter"):
        return True
    if desc.startswith("Complete the final mission"):
        return True
    return desc.startswith("Complete ") and not any(ch.isdigit() for ch in desc)


def dedup_reads(history):
    """Collapse consecutive identical reads to one snapshot each.

    Returns a list of (t_hours, players_tracked, rates tuple), keeping the
    FIRST read of each run: the state is as of Exophase's last refresh
    before that read, and the frozen repeats after it add no information.
    """
    snaps = []
    prev_key = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == prev_key:
            continue
        prev_key = key
        snaps.append((hours_since_release(parse_time(read["slot"])),
                      read["players_tracked"], tuple(read["rates"])))
    return snaps


def refresh_interval_hours(snaps):
    """Typical gap between distinct snapshots: Exophase's refresh cadence."""
    if len(snaps) < 2:
        return 6.0
    gaps = sorted(snaps[i + 1][0] - snaps[i][0] for i in range(len(snaps) - 1))
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return 6.0
    median_gap = gaps[len(gaps) // 2] if len(gaps) % 2 else (
        gaps[len(gaps) // 2 - 1] + gaps[len(gaps) // 2]) / 2.0
    return min(max(median_gap, 0.5), 24.0)


def fit_weights(snaps, t_last, p_last):
    """Recency x precision weights for a fit over snapshots."""
    return [math.exp(-(t_last - t) / RECENCY_TAU_HOURS) * math.sqrt(p / p_last)
            for (t, p, _rates) in snaps]


def baseline_row(rate, horizon_days):
    """Safety-net forecast: persistence plus the house baseline's spread."""
    spread = 0.5 * math.sqrt(max(horizon_days, 0.0))
    lg = logit(rate if rate is not None else 0.0)
    return [round4(inv_logit(lg + norm_ppf(tau) * spread)) for tau in LEVELS]


def quantiles_from(median_pct, sigma, levels):
    """Five quantiles of a logit-normal with the given median and sigma."""
    lg = logit(median_pct)
    out = []
    prev = -1.0
    for tau in levels:
        q = round4(inv_logit(lg + norm_ppf(tau) * sigma))
        q = min(max(q, 0.0), 100.0)
        if q < prev:  # rounding must not break monotonicity
            q = prev
        out.append(q)
        prev = q
    return out


def flat_history_forecast(trophy, last_rate, t_last, horizon_days):
    """Fallback when the history is one frozen page: no trend to fit.

    The next refresh multiplies tracked players several-fold while earners
    grow only a little, so rates fall; story trophies fall least (new
    players earn them fast), platinum most.  Wide spread: the refresh may
    or may not land inside the horizon, and its size is unknown.
    """
    if last_rate is None:
        last_rate = 1.0
    if last_rate >= 99.995:
        return quantiles_from(99.99, 0.15, LEVELS)
    launch = t_last < LAUNCH_WINDOW_HOURS
    if launch:
        p_mult = 4.0
        e_mult = {"platinum": 1.05, "gold": 1.15, "silver": 1.30,
                  "bronze": 2.0 if is_story_trophy(trophy) else 1.8}
        sigma = FLAT_SIGMA_LAUNCH
    else:
        p_mult = 1.5
        e_mult = {"platinum": 1.02, "gold": 1.10, "silver": 1.20,
                  "bronze": 1.5 if is_story_trophy(trophy) else 1.4}
        sigma = FLAT_SIGMA_LATE
    growth = e_mult.get(trophy.get("tier", "bronze"), 1.5) / p_mult
    median = clamp_rate(last_rate * growth)
    return quantiles_from(median, sigma, LEVELS)


def forecast_case(case):
    """Forecast every trophy of one case; always returns a valid object."""
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    levels = case.get("levels") or LEVELS
    history = case.get("history") or []
    if not history:
        return {"trophies": [{"id": t["id"],
                              "q": quantiles_from(1.0, 1.0, levels)}
                             for t in trophies]}

    last_rates = history[-1]["rates"]
    snaps = dedup_reads(history)
    t_last = snaps[-1][0]
    p_last = snaps[-1][1]
    t_target = hours_since_release(parse_time(case["target"]))
    refresh_h = refresh_interval_hours(snaps)
    # The answering read shows the page as of Exophase's last refresh before
    # it: on average half a refresh interval before the target.
    t_eff = t_target - refresh_h / 2.0
    x_target = log_time(t_eff)

    if len(snaps) < 2:
        rows = []
        for i, trophy in enumerate(trophies):
            try:
                q = flat_history_forecast(trophy, last_rates[i], t_last,
                                          horizon_days)
            except Exception:
                q = baseline_row(last_rates[i], horizon_days)
            rows.append({"id": trophy["id"], "q": q})
        return {"trophies": rows}

    # --- fit window: last 48h of distinct snapshots (or all if fewer) ---
    window = [s for s in snaps if s[0] >= t_last - FIT_WINDOW_HOURS]
    if len(window) < 3:
        window = snaps
    w_all = fit_weights(window, t_last, p_last)
    xs = [log_time(t) for (t, _p, _r) in window]

    # --- players-tracked fit: ln P vs log time ---
    p_fit = wls(xs, [math.log(max(p, 1)) for (_t, p, _r) in window], w_all)
    p_target_raw = math.exp(extrapolate(p_fit, x_target, SLOPE_DAMP))
    p_target = min(max(p_target_raw, p_last), p_last * P_GROWTH_CAP)
    sigma_p = p_fit["resid"] * leverage_multiplier(p_fit, x_target)

    # --- per-trophy earner fits: ln E vs log time ---
    n_trophies = len(trophies)
    raw_slopes = [None] * n_trophies
    fits = [None] * n_trophies
    e_last = [None] * n_trophies
    for i in range(n_trophies):
        pts_x, pts_y, pts_w = [], [], []
        for (t, p, rates), w in zip(window, w_all):
            r = rates[i]
            if r is None:
                continue
            earners = max(r * p / 100.0, 0.5)
            pts_x.append(log_time(t))
            pts_y.append(math.log(earners))
            pts_w.append(w)
        if not pts_x:
            continue
        e_last[i] = math.exp(pts_y[-1])
        if len(pts_x) >= 2 and max(pts_y) > min(pts_y) + 1e-9:
            fit = wls(pts_x, pts_y, pts_w)
            fits[i] = fit
            raw_slopes[i] = fit["slope"]
        else:
            fits[i] = {"intercept": pts_y[-1], "slope": 0.0, "resid": 0.0,
                       "n": len(pts_x), "xbar": pts_x[-1], "sxx": 0.0,
                       "sw": sum(pts_w)}
            raw_slopes[i] = 0.0

    # Tier-mean slopes, for shrinkage (weighted by number of fit points).
    tier_sums, tier_ws = {}, {}
    global_sum, global_w = 0.0, 0.0
    for i, slope in enumerate(raw_slopes):
        if slope is None:
            continue
        w = fits[i]["n"]
        tier = trophies[i].get("tier", "bronze")
        tier_sums[tier] = tier_sums.get(tier, 0.0) + w * slope
        tier_ws[tier] = tier_ws.get(tier, 0.0) + w
        global_sum += w * slope
        global_w += w
    global_slope = global_sum / global_w if global_w else 0.0

    # Global residual std across trophies with a real (n >= 3) fit.
    resids = sorted(f["resid"] for f in fits
                    if f is not None and f["n"] >= 3 and f["resid"] > 0)
    if resids:
        sigma_global = resids[len(resids) // 2]
    else:
        sigma_global = DEFAULT_RESID

    # Fallback median rate for a trophy never observed in any read.
    observed_last = [r for r in last_rates if r is not None]
    observed_last.sort()
    median_last = (observed_last[len(observed_last) // 2]
                   if observed_last else 1.0)

    sigma_rw = SIGMA_RW * math.sqrt(max(horizon_days, 0.0))
    rows = []
    for i, trophy in enumerate(trophies):
        try:
            rows.append({"id": trophy["id"],
                         "q": forecast_trophy(
                             i, trophy, last_rates[i], median_last, fits[i],
                             raw_slopes[i], e_last[i], tier_sums, tier_ws,
                             global_slope, sigma_global, sigma_p,
                             p_fit["slope"], p_target, x_target, t_eff,
                             refresh_h, sigma_rw, levels)})
        except Exception:
            rows.append({"id": trophy["id"],
                         "q": baseline_row(last_rates[i], horizon_days)})
    return {"trophies": rows}


def forecast_trophy(i, trophy, last_rate, median_last, fit, raw_slope,
                    e_last_i, tier_sums, tier_ws, global_slope, sigma_global,
                    sigma_p, p_slope, p_target, x_target, t_eff, refresh_h,
                    sigma_rw, levels):
    """One trophy's five quantiles from its earner fit and the P fit."""
    if last_rate is not None and last_rate >= 99.995:
        return quantiles_from(99.99, 0.15, levels)
    if fit is None or e_last_i is None:
        return quantiles_from(median_last, 1.0, levels)

    # Shrink the earner slope toward the tier mean (empirical Bayes).
    tier = trophy.get("tier", "bronze")
    if tier in tier_ws and tier_ws[tier] > 0:
        tier_slope = tier_sums[tier] / tier_ws[tier]
    else:
        tier_slope = global_slope
    n = fit["n"]
    slope = (n * raw_slope + SHRINK_PRIOR_N * tier_slope) / (n + SHRINK_PRIOR_N)

    # Extrapolate earners from the fit's centre with the shrunk, damped
    # slope, with growth caps so a steep log-log line cannot explode over
    # a 4.5-day horizon.
    e_target_raw = math.exp(extrapolate(fit, x_target, SLOPE_DAMP, slope))
    growth = e_target_raw / e_last_i
    growth = min(max(growth, E_GROWTH_FLOOR), E_GROWTH_CAP)
    e_target = e_last_i * growth
    median = 100.0 * e_target / p_target
    median = clamp_rate(median)

    # Guard: the median may not jump implausibly far from the last rate.
    if last_rate is not None:
        lg_last = logit(last_rate)
        lg_med = logit(median)
        if lg_med > lg_last + LOGIT_MOVE_CAP:
            median = inv_logit(lg_last + LOGIT_MOVE_CAP)
        elif lg_med < lg_last - LOGIT_MOVE_CAP:
            median = inv_logit(lg_last - LOGIT_MOVE_CAP)

    # --- spread: fit residual x leverage, P residual, refresh lag, walk ---
    if fit["n"] >= 3 and fit["resid"] > 0:
        resid = math.sqrt((fit["n"] * fit["resid"] ** 2
                           + 4 * sigma_global ** 2) / (fit["n"] + 4))
    else:
        resid = sigma_global
    sigma_fit = resid * leverage_multiplier(fit, x_target)
    # Refresh lag: the answer is as of up to one refresh interval before the
    # target; the logit rate moves at (slope_E - slope_P) / (1 + t) per hour.
    rate_of_change = abs(slope - p_slope) / (1.0 + max(t_eff, 0.0))
    sigma_refresh = rate_of_change * refresh_h / math.sqrt(12.0)
    sigma = math.sqrt(sigma_fit ** 2 + sigma_p ** 2
                      + sigma_refresh ** 2 + sigma_rw ** 2)
    sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
    return quantiles_from(median, sigma, levels)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            # Last-resort safety net: persistence for every trophy.
            horizon_days = case.get("horizon_minutes", 0) / 1440.0
            history = case.get("history") or []
            last_rates = history[-1]["rates"] if history else []
            rows = []
            for idx, trophy in enumerate(case["trophies"]):
                rate = last_rates[idx] if idx < len(last_rates) else None
                rows.append({"id": trophy["id"],
                             "q": baseline_row(rate, horizon_days)})
            forecast = {"trophies": rows}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
