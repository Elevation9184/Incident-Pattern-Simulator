# Incident Pattern Simulator + Incident Overlap 3D Explorer
**Making roster-overlap statistics legible**

This repository contains two linked Monte Carlo tools for explaining why a nurse can appear at many incidents under neutral staffing conditions. The current build now incorporates the main structural point raised by Prof Richard Gill: nurse presences are not fully independent across shifts, and day and night staffing should be treated as distinct regimes rather than as one exchangeable pool.

- `Incident_Pattern_Simulator.html` explores individual scenarios in the browser.
- `incident_overlap_3D_generator.py` generates exported manifold packs for the 3D explorer.
- `incident_overlap_3D_explorer.html` visualises the exported null surfaces and compares an observed overlap against them.
- `Incident_Roster_Analysis.docx` provides the companion accessible + technical documentation.

## Current Model Status

The project now supports three staffing models across the simulator and generator:

- `unconstrained`: the original independent workload model
- `fixed`: each shift draws an exact headcount without replacement
- `day_night`: separate day and night staffing regimes with different fixed shift sizes

The generator always exports the combined pack `manifold_roster_pack.json`. When running in `day_night` staffing mode (the Gill-driven addition), it also exports regime-specific packs:

- `manifold_roster_pack_day.json`
- `manifold_roster_pack_night.json`

The explorer reads the `staffing_mode` from the combined pack and only shows the Day/Night regime toggle when the packs were generated under `day_night` mode. In `unconstrained` or `fixed` mode there is no meaningful day/night distinction, so the toggle is hidden.

Each pack keeps the same surface schema and also exports the simulated max-overlap histogram for every `(M, p-bar)` cell. That lets the explorer show exact percentile and tail-probability views rather than only mean / Q95 / Q99 summaries.

## Explorer Features

The current `incident_overlap_3D_explorer.html` includes:

- three surfaces: `Mean`, `Q95`, `Q99`
- observed-overlap comparison with continuous traffic-light coloring
- `All shifts / Day / Night` regime toggle with animated surface transitions (visible only in `day_night` staffing mode)
- `M-SLICE PROFILE` panel for the current incident-count slice
- profile toggle between raw overlap and tail probability
- plain-English rarity readout for the observed count
- fixed-ward vs ensemble-roster metadata labels

Interpretation note: raw day and night surfaces need not rank the same way as day and night rarity. The more Gill-aligned question is usually the tail question: how often would chance alone produce at least this overlap under the relevant staffing regime?

## Simulator Features

The browser simulator keeps the original scenario-level analysis modes and now adds staffing-constraint controls under advanced configuration:

- `Staffing Constraint Mode`
- `Nurses per shift` for fixed-headcount simulations
- `Day shift staffing`
- `Night shift staffing`
- `Day/Night ratio`

This lets the single-scenario tool and the exported 3D manifolds tell the same story.

## Default Case Parameters

The repository still defaults to the Letby-style reference values:

- `S = 730` shifts
- `N = 38` nurses
- `beta_a = 3.2`
- `beta_b = 17.25`
- `intercept_M = 61`
- `intercept_mean_presence = 0.204`

## Regenerating The Manifolds

Run:

```bash
python incident_overlap_3D_generator.py
```

The generator will export the combined pack, plus day and night packs when `staffing_mode = "day_night"` and `export_regime_packs = True`. The explorer expects the JSON files to live alongside `incident_overlap_3D_explorer.html`.

## What The Tools Are For

These tools are designed to make the statistical point accessible:

- high overlap can arise from ordinary workload differences
- fixed staffing reduces variance relative to the looser independent model
- day and night should be judged against different neutral benchmarks
- "rare under innocence" is not the same thing as "evidence of guilt"

## Limitations

This is an educational and explanatory project, not a forensic instrument.

- incidents are still placed uniformly at random unless you add further structure
- full rota rules such as leave, consecutive nights, and skill-mix constraints are not modelled
- acuity confounding is simplified rather than reconstructed from case data
- any live-case use would require independent expert review

## Documentation

- Companion guide: [Incident_Roster_Analysis.docx](Incident_Roster_Analysis.docx)
- Developer/status note: [CLAUDE.md](CLAUDE.md)

## References

- O'Quigley, J. (2025). *Use of roster charts in the investigation and prosecution of nurses suspected of inflicting deliberate harm on patients.*
- O'Quigley, J. (2025). *Suspected serial killers and unsuspected statistical blunders.*
- Richard Gill's March 2026 walk-throughs and RPubs notes on the beta-binomial critique and constrained staffing

---

*"Someone will always be the maximum. The question is how often chance alone would make that happen."*
