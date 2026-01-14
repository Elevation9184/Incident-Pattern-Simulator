# Incident Pattern Simulator

**The statistics of being in the wrong place at the wrong time**

*Beta Release*

An interactive simulator exploring the statistical foundations of "incident overlap" evidence in healthcare settings. Built as a pedagogical tool to help lawyers, journalists, statisticians, and the public understand why a nurse present at many incidents isn't necessarily evidence of wrongdoing.

## The Core Problem

When a nurse works more shifts than average, they will naturally be present at more incidents—simply because they were there more often. This simulator demonstrates that apparent "suspicious coincidence" often emerges from ordinary workload patterns, not wrongdoing.

## What This Tool Does

The simulator runs thousands of Monte Carlo trials where:
- Incidents are placed randomly across shifts
- Nurses' schedules are drawn from realistic workload distributions  
- Overlap counts reveal how often the hardest-working nurse appears "suspicious"

**Key insight:** The nurse who works the most will almost always have the highest overlap—regardless of guilt.

## Try It

**[Launch the Pattern Simulator](Incident_Pattern_Simulator.html)** — Explore individual scenarios

**[Launch the Overlap Tracker](incident_overlap_tracker.html)** — Explore the statistical manifold

![Monte Carlo Manifold Surfaces](manifold_3d_surfaces_R5000_ensemble_rosters.png) 

## Features

### Three Analysis Modes

| Mode | What It Tests |
|------|---------------|
| **Standard** | Is this specific nurse's overlap unusual given their workload? |
| **Selection Effect** | How does investigator bias inflate apparent coincidence? |
| **No Pre-designation** | What's the expected maximum overlap when nobody is pre-selected? |

### Statistical Framework

- **Hypergeometric distribution** for exact shift-based overlap
- **Beta-binomial workload model** with intuitive Mean/Consistency parameters
- **Monte Carlo confidence intervals** for key statistics
- **3D manifold surfaces** from 5,000 trials per (M, p̄) grid cell
- Ward staffing presets including parameters from O'Quigley (2025)

### Overlap Manifold Explorer

Interactive 3D visualisation of how maximum incident overlap scales under the null hypothesis:

- **Three statistical surfaces:** Mean, Q95, Q99 percentiles across 5,000 Monte Carlo trials per grid point
- **Two-parameter exploration:** Number of incidents (M) vs mean shift density (p̄)
- **Real-time interpolation:** Drag the intercept marker to query any point on the manifold
- **Grounded baselines:** What overlap counts are expected by chance alone?

The manifold data can be regenerated using `incident_overlap_generator.py` with custom parameters.

### Judicial Commentary

Side-by-side prosecution and defence framings of the same statistics, demonstrating:
- The prosecutor's fallacy (confusing P(evidence|innocence) with P(innocence|evidence))
- Why "present at 92% of incidents" means nothing without workload context
- The gap between mean outcomes and extreme peaks

## Important Limitations

⚠️ **Educational Purpose Only**

This simulator is a teaching aid, not a forensic instrument. The models use simplified assumptions (uniform incident distribution, exchangeable shift probabilities) that may not hold in specific cases.

**This tool must not be:**
- Submitted as evidence in court
- Used to argue guilt or innocence
- Applied to real case data without qualified expert oversight

Any application to live cases requires verification by a Chartered Statistician (CStat, RSS), Accredited Professional Statistician (PStat, ASA), or equivalent.

## Background Reading

- O'Quigley, J. (2025). "Use of roster charts in the investigation and prosecution of nurses suspected of inflicting deliberate harm on patients." *Medicine, Science and the Law*
- O'Quigley, J. (2025). "Suspected serial killers and unsuspected statistical blunders." *Medicine, Science and the Law*, 65: 28–35
- Meester et al. — On the (ab)use of statistics in medical-legal cases
- Thompson & Schumann — Interpretation of statistical evidence in criminal trials

## Documentation

Full technical documentation including mathematical framework, parameter explanations, and interpretation guidance is available in [Incident_Pattern_Simulator.docx](Incident_Pattern_Simulator.docx).

The manifold generator source (`incident_overlap_generator.py`) includes detailed inline documentation of the Monte Carlo methodology.

## Licence

MIT License — free to use, adapt, and share.

---

*"Someone will always be the maximum. Being the maximum isn't evidence of guilt."*
