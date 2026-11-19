#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster — make.fun bounty gta6-trophy-rarity-forecast-01.

Rebuilt in this session from the method specification (the reference
session's REPORT.md / README.md and the gta6_trophy_forecast skill),
written from scratch in my own words and structure. Python standard
library only, no network, no randomness: the same case always yields
the same bytes.

The one fact the whole design turns on: Exophase earn rates fall as
the tracked-player denominator grows, and the page refreshes in steps
rather than smoothly. The one refresh observed so far moved 54 of 55
rates down by roughly 1.5-1.9 logit units with no warning, while every
history read before it was byte-identical. So a flat history must not
earn a narrow spread: the forecaster works in logit space, fits two
damped trend curves (against log-time since release and against log
players-tracked), and carries a wide refresh-shock spread whenever the
history has never moved.

Usage: python3 main.py — reads $MAKEFUN_INPUT_DIR (default /input),
writes one <id>.out per <id>.in to $MAKEFUN_OUTPUT_DIR (default
/output). Output: {"trophies": [{"id": n, "q": [q05, q25, q50, q75,
q95]}, ...]}, each q a finite number in [0, 100], non-decreasing.
"""

import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
NORM_Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
          0.6744897501960817, 1.6448536269514722)

# Tuned constants (from the spec; the shock value was chosen by a
# sweep on the synthetic shock test, 0.6 -> 1.136 ... 1.5 -> 0.861).
SHOCK_FLAT = 1.20        # logit spread when the history never moved
SHOCK_LO = 0.25          # clip for the step-based shock estimate
SHOCK_HI = 1.00
SIGMA_LO = 0.30
SIGMA_HI = 2.20
MAX_MEDIAN_MOVE = 3.5    # logit cap on the median's move from last rate
LONG_HORIZON_H = 24.0
PLAYERS_CAP_MULT = 50.0  # players at target, at most 50x the last read


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _clamp_rate(rate):
    return min(99.99, max(0.01, rate))


def _logit(pct):
    p = _clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def _sigmoid(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def _round4(value):
    """Round to 4 decimals, half-up (so 0.12345 -> 0.1235, not banker's)."""
    q = Decimal(repr(value)).quantize(Decimal("0.0001"),
                                      rounding=ROUND_HALF_UP)
    return float(q)


def _inv_normal(p):
    """Acklam's rational approximation to the standard normal quantile."""
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
               + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if p <= phigh:
        q = p - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
               + a[5]) * q
        den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r
               + 1.0)
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
           + c[5])
    den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    return -num / den


def _wls(xs, ys, ws):
    """Weighted least-squares line, with the pieces a prediction
    interval needs: slope, intercept, weighted mean x, Sxx, and a
    residual-variance estimate."""
    sw = sum(ws)
    xm = sum(w * x for w, x in zip(ws, xs)) / sw
    ym = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xm) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - xm) * (y - ym) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0
    intercept = ym - slope * xm
    n = len(xs)
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = n - 2
    if dof > 0:
        var = sum(w * r * r for w, r in zip(ws, resid)) / (sw * dof / n)
    else:
        var = 0.0
    return {"slope": slope, "intercept": intercept, "xm": xm, "ym": ym,
            "sxx": sxx, "resid_var": max(var, 0.0), "n": n}


def _shrunk_at(fit, shrink, x_target):
    """Line value at x_target with the slope shrunk toward 0, anchored
    at the weighted mean (xm, ym) so shrinking flattens the line
    around the data's centre instead of shifting it."""
    return fit["ym"] + shrink * fit["slope"] * (x_target - fit["xm"])


def _pred_se(fit, x_target):
    """Prediction standard error of a fitted line at x_target."""
    n = fit["n"]
    if fit["resid_var"] <= 0:
        return 0.0
    lever = 1.0 + 1.0 / n
    if fit["sxx"] > 0:
        lever += (x_target - fit["xm"]) ** 2 / fit["sxx"]
    return math.sqrt(fit["resid_var"] * lever)


def _distinct_reads(history, idx):
    """Collapse the history to distinct reads: a run of identical
    (rate, players_tracked) counts once, at its last time. Returns a
    list of (hours_since_release, players_tracked, rate), oldest first.
    A run of identical reads carries no new information and must not
    dominate the fit."""
    obs = []
    for row in history:
        rate = row["rates"][idx]
        if rate is None:
            continue
        at = _parse_time(row["at"])
        hours = (at - RELEASE).total_seconds() / 3600.0
        players = row["players_tracked"]
        if obs and obs[-1][1] == players and obs[-1][2] == rate:
            obs[-1] = (hours, players, rate)
        else:
            obs.append((hours, players, rate))
    return obs


def _quantiles(median_logit, sigma):
    qs = []
    for z in NORM_Z:
        qs.append(_round4(_sigmoid(median_logit + z * sigma)))
    for k in range(1, 5):
        qs[k] = max(qs[k], qs[k - 1])
    return qs


def _forecast_one(obs, horizon_hours, target):
    n = len(obs)
    if n == 0:
        # No read ever listed this trophy: a wide, centred guess.
        return _quantiles(_logit(1.0), SIGMA_HI)
    last_rate = obs[-1][2]
    last_logit = _logit(last_rate)

    # A trophy at 100% in *every* read stays at 100%: it survived
    # every refresh so far at the ceiling.
    if all(rate >= 100.0 for _, _, rate in obs):
        return _quantiles(_logit(99.99), 0.35)

    horizon_days = horizon_hours / 24.0
    walk = 0.5 * math.sqrt(horizon_days)

    rates_seen = {rate for _, _, rate in obs}
    never_moved = len(rates_seen) == 1

    if n == 1 or never_moved:
        # One distinct read (or several identical ones): no trend can
        # be estimated; the median stays at the last rate and the
        # spread is dominated by the refresh shock.
        median = last_logit
        trend_se = 0.0
        shock = SHOCK_FLAT
    else:
        span = obs[-1][0] - obs[0][0]
        shrink = (n / (n + 2.0)) * min(1.0, span / 12.0)
        # Recency weights: 1.0 for the oldest read, 2.0 for the newest.
        if span > 0:
            ws = [1.0 + (h - obs[0][0]) / span for h, _, _ in obs]
        else:
            ws = [1.0] * n
        log_hours = [math.log1p(max(0.0, h)) for h, _, _ in obs]
        logits = [_logit(rate) for _, _, rate in obs]

        # Trend (a): launch-cohort curve, logit(rate) vs log-time.
        fit_a = _wls(log_hours, logits, ws)
        target_hours = (target - RELEASE).total_seconds() / 3600.0
        x_a = math.log1p(max(0.0, target_hours))
        med_a = _shrunk_at(fit_a, shrink, x_a)

        # Trend (b): dilution curve, logit(rate) vs log players.
        # Players at the target come from a power law in log-time:
        # log players linear in log1p(hours), slope floored at 0
        # (the tracked base only grows), capped at 50x the last read.
        log_players = [math.log(p) for _, p, _ in obs]
        fit_p = _wls(log_hours, log_players, ws)
        slope_p = max(0.0, fit_p["slope"])
        players_target = math.exp(fit_p["intercept"] + slope_p * x_a)
        players_target = min(players_target,
                             PLAYERS_CAP_MULT * obs[-1][1])
        players_target = max(players_target, obs[-1][1])
        fit_b = _wls(log_players, logits, ws)
        x_b = math.log(players_target)
        med_b = _shrunk_at(fit_b, shrink, x_b)

        median = 0.5 * (med_a + med_b)
        move = median - last_logit
        if abs(move) > MAX_MEDIAN_MOVE:
            median = last_logit + math.copysign(MAX_MEDIAN_MOVE, move)

        # Trend spread: mean prediction error of the two fits,
        # damped when the horizon is short against the span observed.
        se = 0.5 * (_pred_se(fit_a, x_a) + _pred_se(fit_b, x_b))
        trend_se = se * min(1.0, horizon_hours / max(span, 1.0))

        # Refresh shock from the steps actually seen: RMS of the
        # logit moves between successive distinct reads.
        steps = [_logit(obs[i + 1][2]) - _logit(obs[i][2])
                 for i in range(n - 1)]
        rms = math.sqrt(sum(s * s for s in steps) / len(steps))
        shock = min(SHOCK_HI, max(SHOCK_LO, rms))

    if never_moved and horizon_hours > LONG_HORIZON_H:
        # The one refresh observed so far moved 54/55 rates down:
        # on a long horizon a flat history is expected to fall.
        median -= 0.30

    sigma = math.sqrt(walk * walk + trend_se * trend_se + shock * shock)
    sigma = min(SIGMA_HI, max(SIGMA_LO, sigma))
    return _quantiles(median, sigma)


def forecast_case(case):
    target = _parse_time(case["target"])
    horizon_hours = case["horizon_minutes"] / 60.0
    rows = []
    for idx, trophy in enumerate(case["trophies"]):
        obs = _distinct_reads(case["history"], idx)
        rows.append({"id": trophy["id"],
                     "q": _forecast_one(obs, horizon_hours, target)})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
