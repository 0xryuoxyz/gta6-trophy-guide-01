# GTA VI Trophy Rarity Forecaster (rebuild)

Entry for make.fun bounty `gta6-trophy-rarity-forecast-01`:
forecast, for each of the game's 55 PS5 trophies, five quantiles
(5%, 25%, 50%, 75%, 95%) of its Exophase earn rate at 2026-11-26T00:00:00Z.

Rebuilt 19 Nov 2026 in a session started after the bounty opened
(11:30 UTC), from the method specification in `../REPORT.md` and the
`gta6_trophy_forecast` skill — written from scratch, not copied from
the reference implementation in `../entry/`.

## Run

```sh
python3 main.py          # reads $MAKEFUN_INPUT_DIR (default /input),
                         # writes $MAKEFUN_OUTPUT_DIR (default /output)
```

Python 3.13, standard library only, no network, deterministic,
~0.03 s per case (limit: 8 s, 1 CPU, 1024 MB).

## Method

Per trophy, in logit space (rates clamped to 0.01..99.99):

1. Collapse the history to distinct reads — Exophase refreshes on its
   own schedule, so runs of identical reads carry no new information.
2. Two weighted-least-squares trend extrapolations, averaged in logit
   space: (a) logit(rate) vs ln(1 + hours since release), the
   launch-cohort curve; (b) logit(rate) vs ln(players_tracked), the
   dilution curve, with players_tracked itself extrapolated by a
   power law in log-time. Slopes are shrunk toward 0 when they rest on
   few distinct reads or a short span, anchored at the weighted mean;
   the median's move from the last read is capped at 3.5 logit units.
3. Spread = random walk (0.5 * sqrt(horizon days)) combined in
   quadrature with the trend fit's residual prediction spread and a
   refresh-shock term: 1.20 logit when the history has never moved
   (the one refresh observed so far moved 54 of 55 rates by ~1.5-1.9
   logit units without warning), otherwise the pooled RMS of the
   logit steps between successive distinct reads (clipped 0.25..1.0).
   Sigma is clipped to [0.30, 2.20].
4. A trophy at 100% in *every* read is forecast to stay at 99.99%
   with a small spread. A flat-history trophy on a horizon over 1 day
   gets a -0.30 logit median shift, because the observed refresh
   moved rates down (player-base growth outpaces earners).

See `REPORT-REBUILD.md` for the test results and what is still
missing before submission.
