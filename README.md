# GTA VI Trophy Rarity Forecast — Week One

[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/erlansyahm90-code/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01/)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/erlansyahm90-code/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01/)

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
([competition page](https://make.fun/c/gta6-trophy-rarity-forecast-01/)):
forecast every PlayStation 5 trophy of Grand Theft Auto VI's earn rate on
[Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) per trophy,
scored by mean pinball loss on the logit scale.

## The model

An earn rate is a ratio — `earners / players_tracked` — and over launch week
both parts move fast for different reasons. Fitting the rate directly
confuses the two, so `main.py` (Python 3.13, standard library only) fits each
side separately in log space against log time since release, then divides:

```
E_i(t) ~ exp(a_i + b_i * ln(1 + t))     earners of trophy i
P(t)   ~ exp(a_P + b_P * ln(1 + t))     players tracked
r_i    = 100 * E_i(target) / P(target)
```

This separates **dilution** (tracked players jump, so rates of trophies the
newcomers haven't earned fall — the house saw exactly this on launch night:
one refresh took players 638 → 3,868 and every rate but the 100% one fell)
from **progression** (earners grow as the tracked base plays: story trophies
fast, grind/collectible/platinum slowly).

What makes it beat the baselines:

- **Refresh-aware reads.** Exophase refreshes on its own schedule, so the
  house's reads repeat the same page for hours. Consecutive identical reads
  are collapsed to one snapshot before fitting; otherwise the fit is
  dominated by frozen plateaus.
- **Recent, recency-weighted fits.** Weighted least squares over the last
  24h of distinct snapshots (tau = 24h, plus a √P precision weight). Launch
  growth decelerates, so the recent slope is the one that extrapolates —
  24h clearly beats 48h in backtest.
- **Tier shrinkage.** Each trophy's earner slope is shrunk toward its
  tier's mean slope (empirical Bayes), so trophies with few distinct reads
  borrow strength instead of extrapolating noise.
- **Damped extrapolation.** Slopes are damped to 80% around the fit's
  centre: growth keeps decelerating past the fit window, so the undamped
  line overshoots the 4.5-day target.
- **Refresh-lag-aware target.** The answering read shows the page as of
  Exophase's last refresh before it, so the forecast is evaluated at
  target minus half the typical refresh interval (estimated from the gaps
  between distinct snapshots).
- **Honest spread.** Fit residual × leverage, players-fit residual,
  refresh-lag uncertainty, and a random-walk term growing with the horizon;
  floored at 0.20 and capped at 1.50 logit units.
- **Frozen-page fallback.** When the history is a single frozen page (the
  practice cases), there is no trend to fit, so the launch-night dilution
  prior applies: the next refresh multiplies tracked players several-fold
  while earners grow only a little, so rates fall, with a wide spread. A
  trophy at 100% stays (near) 100% — it survived the 6x refresh at 100%.
- **Never invalid.** Any per-trophy or whole-case failure falls back to
  persistence plus a random-walk spread, so the program always writes a
  valid forecast file.

Deterministic (pure float arithmetic, no randomness, files in name order),
~0.08 s per case, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes `<id>.out` per case.

## Backtest

Practice answers are not in the kit, so the model was validated two ways
(scripts in `/tmp/forecast-test`, not shipped):

- **Refresh-proxy backtest** on the practice cases, using the refresh the
  rules disclose (players 638 → 3,868, every rate but the 100% one lower;
  proxy truth = earners frozen): **0.51** vs house baseline 1.69.
- **Synthetic launch-week simulation** with known truth — a cohort
  convolution model (players join on a Weibull curve; each trophy has a
  lognormal time-to-earn solved from plausible deadline/target rates),
  Exophase-style refresh jumps, 8 seeds × 3 speeds. Mean pinball loss:

  | world | this model | house baseline | house curve baseline |
  |-------|-----------:|---------------:|---------------------:|
  | normal | **0.17** | 0.63 | 0.33 |
  | slow | **0.16** | 0.62 | 0.30 |
  | fast | **0.23** | 0.65 | 0.39 |

  24/24 seed wins against both house entries. (Simulation evidence, not a
  guarantee — the real week-one dynamics are the judge.)

## Files

- `main.py` — the whole entry: model, forecast, I/O.
- `LICENSE` — MIT.
