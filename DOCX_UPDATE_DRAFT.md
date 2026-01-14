# DOCX Update Draft: Incident Overlap Tracker

*This markdown content is intended to be added to Incident_Pattern_Simulator.docx as a new section.*

---

## Overlap Manifold Explorer

### Purpose

The Overlap Tracker provides a complementary perspective to the Pattern Simulator. While the Pattern Simulator explores individual scenarios with specific parameters, the Overlap Tracker visualises the **full statistical landscape** of incident-roster overlap under the null hypothesis of random incident occurrence.

This answers a broader question: across all reasonable combinations of incident count (M) and mean shift exposure (p̄), what maximum overlap values should we expect by chance alone?

### Monte Carlo Methodology

The manifold surfaces are generated through extensive Monte Carlo simulation:

**Simulation Parameters:**
- **S = 730 shifts** (approximately two years of daily assignments)
- **N = 38 nurses** on the ward roster
- **R = 5,000 trials** per grid cell (880 grid cells total)
- **Workload distribution:** Beta(3.2, 17.25), rescaled to mean presence p̄
- **Presence clipping:** Individual nurse presence capped at 95% of shifts

**Grid Structure:**
- **Incident axis (M):** 1 to 80 incidents in unit steps
- **Mean presence axis (p̄):** 0.10 to 0.40 in 11 equally-spaced intervals
- **Total grid:** 11 × 80 = 880 parameter combinations

**Ensemble Rosters Mode:**
Each Monte Carlo trial independently redraws nurse workloads from the Beta distribution. This models the realistic scenario where the "high-workload" role rotates among staff over time, introducing appropriate variability into the null distribution.

### Statistical Surfaces Explained

For each grid cell (M, p̄), the simulation records the **maximum overlap** across all N nurses in each of R trials. Three summary statistics are then computed:

| Surface | Definition | Interpretation |
|---------|------------|----------------|
| **Mean** | Average of max overlaps across 5,000 trials | Typical expected maximum under neutrality |
| **Q95** | 95th percentile (1-in-20 threshold) | "Elevated but not rare" maximum |
| **Q99** | 99th percentile (1-in-100 threshold) | "Rare but plausible by chance" maximum |

These surfaces quantify the **baseline expectation** against which observed overlap values can be contextualised. An observed overlap near or below the mean surface is unremarkable; one exceeding Q99 warrants further investigation but remains statistically possible under the null hypothesis.

### Using the 3D Tracker

**Interface Elements:**

1. **Left Panel Controls**
   - *Incidents slider:* Set the number of incidents (M) from 1 to 80
   - *Mean presence slider:* Set shift exposure (p̄) from 0.10 to 0.40
   - *Intercept readout:* Displays interpolated Mean, Q95, Q99 values at current position

2. **3D Scene**
   - Three semi-transparent surfaces (blue = Mean, orange = Q95, green = Q99)
   - Floor grid showing the M × p̄ parameter space
   - Black X marker indicating the current intercept position
   - Coloured spheres showing where the intercept intersects each surface

3. **Interactive Navigation**
   - *Drag scene:* Rotate the 3D view (orbit controls)
   - *Scroll:* Zoom in/out
   - *Drag X marker:* Move the intercept point directly on the floor grid

**Reading the Display:**

Position the intercept at the (M, p̄) values matching a case of interest. The three surface intercepts then show:
- What maximum overlap is **typical** (Mean)
- What maximum overlap is **elevated but not uncommon** (Q95)
- What maximum overlap is **rare but possible by chance** (Q99)

### Regenerating the Manifold

The Python generator (`incident_overlap_generator.py`) can regenerate manifold data with custom parameters:

```python
# Key configuration options in the script:

S = 730           # Total shifts
N = 38            # Number of nurses
R = 5000          # Monte Carlo trials per grid cell
beta_a = 3.2      # Workload Beta distribution shape
beta_b = 17.25    # Workload Beta distribution shape
ensemble_rosters = True   # Redraw workloads each trial

# To generate with export:
export = True     # Outputs JSON manifold pack
```

**Output Files:**
- `manifold_3d_surfaces_R{R}_{mode}.png` — 3D visualisation of all three surfaces
- `manifold_pack_{mode}_R{R}_{timestamp}.json` — Compact JSON for the HTML viewer

The JSON format stores all three surfaces as flattened arrays (row-major order) along with full metadata for reproducibility.

### Interpreting Results

**What the manifold shows:**
- How maximum overlap **scales** with incident count and shift exposure
- The **spread** between typical (Mean) and extreme (Q99) outcomes
- Where in parameter space overlap values become diagnostically meaningful

**What the manifold does NOT show:**
- Guilt or innocence
- Whether a specific overlap is "suspicious"
- Adjustment for non-uniform incident timing or clustering

**Appropriate use:**
The manifold provides statistical context, not conclusions. An observed overlap exceeding Q99 indicates the value is unusual under the null hypothesis, but does not establish causation. Conversely, an overlap within the Q95 envelope should prompt extreme caution before drawing any adverse inference.

### Relationship to Pattern Simulator

| Aspect | Pattern Simulator | Overlap Tracker |
|--------|-------------------|-----------------|
| **Focus** | Single scenario, detailed analysis | Full parameter space, overview |
| **Output** | Distributions, p-values, framings | 3D surfaces, intercept values |
| **Question** | "Is this specific overlap unusual?" | "What overlaps are expected by chance?" |
| **Use case** | Case-specific investigation | Baseline establishment, context setting |

Together, these tools enable a comprehensive statistical assessment: the Overlap Tracker establishes the null landscape, while the Pattern Simulator examines specific observations within that context.

---

*End of proposed DOCX addition*
