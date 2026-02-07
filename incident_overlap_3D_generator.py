"""Incident–Roster Dynamics: Manifold / Surface Visualisation (Numba-optimised)

Generates 3D semi-transparent surfaces and 2D heatmaps showing how the
maximum incident–roster overlap (across nurses) scales with:
  - number of incidents (M)
  - mean shift density / presence fraction (p̄)

It also optionally computes "multi-suspect" metrics:
  - E[# nurses with overlap >= T]
  - P(# nurses with overlap >= T >= 2)

Two modelling modes:
  1) ensemble_rosters=True: re-draw workloads each trial ("high-workload role" rotates)
     → Answers: "What surface stats might ANY nurse encounter across many possible rosters?"
     → Independent sampling per cell; double-averaging produces naturally smooth surfaces.
     → Optimisation: Per (p̄, trial) we draw workloads once and reuse an incident-prefix path 
       across all M values; this preserves the correct per-cell marginals while greatly 
       improving runtime.
  
  2) ensemble_rosters=False: fixed ward structure with scaled workloads
     → Draws ONE set of relative workloads (who works more/less than average)
     → Scales this structure to each mean presence level (same relative patterns, different totals)
     → Each nurse has a fixed "preference ordering" of shifts they work
     → For each trial and incident count M, the SAME incident positions are used across
       all mean_presence values (row-consistent incidents)
     → Answers: "For this specific ward structure, how would overlap statistics change
       if average presence were different, or if different numbers of incidents occurred?"
     → Produces smooth surfaces that vary monotonically along both axes

Dependencies:
  - numpy
  - matplotlib
  - numba

Run:
  python incident_overlap_3D_generator.py

Notes:
  - We implement Beta(a,b) via Gamma draws (Numba-friendly).
  - For ensemble_rosters=False, the key insight is that workloads are SCALED rather than
    independently drawn at each presence level. A nurse's schedule at p=0.20 is a subset
    of their schedule at p=0.30 (they work their "most preferred" shifts first).
  - For very large grids / R, consider reducing grid density or using a coarser
    incident grid, then refining around regions of interest.
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass
from typing import Tuple

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from numba import njit
except Exception as e:
    raise ImportError(
        "Numba is required for this script. Install with: pip install numba"
    ) from e


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class Config:
    # Core roster geometry
    S: int = 730          # shifts
    N: int = 38           # nurses

    # Workload distribution (shape only) for relative presence heterogeneity
    beta_a: float = 3.2
    beta_b: float = 17.25

    # Grids
    incident_grid: Tuple[int, ...] = tuple(range(1, 81, 1))  # M values (unit-level granularity)
    mean_presence_grid: Tuple[float, ...] = tuple(np.linspace(0.10, 0.40, 11))

    # Monte Carlo trials
    R: int = 5000  # number of trials per cell
    
    # Progress reporting
    progress_interval: int = 50  # report progress every N cells (0 = no progress)

    # Threshold metrics
    threshold_T: int = 20

    # Model choice
    ensemble_rosters: bool = True   # True = redraw workloads each trial

    # Presence scaling / realism
    presence_clip_max: float = 0.95  # avoid >1 presence after rescaling tail
    # Quantiles for tail surfaces (interpretable as 1-in-k events)
    q95: float = 0.95
    q99: float = 0.99

    # Randomness
    seed: int = 7

    # Plotting
    show_3d: bool = True

    # Threshold heat-maps (optional diagnostic)
    # ---------------------------------------
    # In addition to the max-overlap surfaces (Mean/Q95/Q99), the generator can compute
    # threshold-based maps that answer two practical questions under the null model:
    #
    #   1) E[count nurses with overlap >= T]  : how many "high-overlap" nurses you should
    #      expect by chance at each (M, p̄). This illustrates why multiple "suspicious"
    #      nurses are often expected when workloads are uneven (a multiple-comparisons effect).
    #
    #   2) P(at least 2 nurses with overlap >= T): how often you get more than one apparent
    #      outlier. This is useful for sanity-checking narratives that treat a single high
    #      overlap as uniquely incriminating.
    #
    # These diagnostics are intentionally outside the main UI/docs scope, but they’re handy
    # for deeper sensitivity analysis when choosing or debating a “concerning” threshold T.
    show_heatmaps: bool = False

    # Intercept annotation (e.g., Letby-style "suspicious incidents" + estimated mean presence)
    show_intercept: bool = True
    intercept_M: float = 61.0
    intercept_mean_presence: float = 0.204       # 0.22
    intercept_label: str = f"Intercept $\\mathbf{{x}}$ (M=61, p̄=0.22)"
    annotate_intercept_on_heatmaps: bool = False

    # Saving
    save_figures: bool = True
    save_format: str = "png"
    save_dpi: int = 200
    file_prefix: str = "manifold"

    # Data export (for JS viewer / later reuse)
    export: bool = True
    export_json_stem: str = "manifold_pack"
    export_pretty: bool = False

    # Heatmap rendering
    heatmap_interpolation: str = "nearest"

    def get_incident_grid(self) -> Tuple[int, ...]:
        """Return incident grid based on export setting."""
        if self.export:
            return tuple(range(1, 81, 1))  # Full detail: 80 points
        else:
            return tuple(range(10, 81, 10))  # Coarse: 8 points


# -----------------------------
# Numba-optimised RNG helpers
# -----------------------------

@njit(cache=True)
def beta_via_gamma(a: float, b: float) -> float:
    """Draw from Beta(a,b) using two Gamma draws (Numba-friendly)."""
    x = np.random.gamma(a, 1.0)
    y = np.random.gamma(b, 1.0)
    return x / (x + y)


@njit(cache=True)
def hypergeom_sequential(ngood: int, nbad: int, nsample: int) -> int:
    """Exact hypergeometric draw via sequential without-replacement sampling."""
    good = ngood
    bad = nbad
    total = good + bad
    succ = 0

    if nsample > total:
        nsample = total

    for _ in range(nsample):
        if total <= 0:
            break
        if good > 0:
            if np.random.random() < (good / total):
                succ += 1
                good -= 1
            else:
                bad -= 1
        else:
            bad -= 1
        total -= 1

    return succ


# -----------------------------
# Helpers (ADD THESE)
# -----------------------------

@njit(cache=True)
def _argsort_int64(a: np.ndarray) -> np.ndarray:
    """Numba-friendly argsort for small int arrays."""
    J = a.shape[0]
    idx = np.empty(J, dtype=np.int64)
    for i in range(J):
        idx[i] = i
    # selection sort (fine for small J)
    for i in range(J - 1):
        min_k = i
        min_v = int(a[idx[i]])
        for j in range(i + 1, J):
            v = int(a[idx[j]])
            if v < min_v:
                min_v = v
                min_k = j
        if min_k != i:
            tmp = idx[i]
            idx[i] = idx[min_k]
            idx[min_k] = tmp
    return idx


@njit(cache=True)
def _build_rank_from_pref(nurse_pref_orders: np.ndarray) -> np.ndarray:
    """rank[n, shift] = position of shift in nurse_pref_orders[n, :]."""
    N, S = nurse_pref_orders.shape
    rank = np.empty((N, S), dtype=np.int64)
    for n in range(N):
        for pos in range(S):
            shift = nurse_pref_orders[n, pos]
            rank[n, shift] = pos
    return rank


@njit(cache=True)
def _hypergeom_prefix_counts(ngood: int, nbad: int, nsample_max: int, out: np.ndarray) -> None:
    """
    Fill out[m] = #successes after (m+1) sequential hypergeometric draws.
    Equivalent marginally to hypergeom_sequential(ngood, nbad, M) for each M,
    but computed in one pass up to nsample_max.
    """
    good = ngood
    bad = nbad
    total = good + bad
    succ = 0

    if nsample_max > total:
        nsample_max = total

    for m in range(nsample_max):
        if total <= 0:
            out[m] = succ
            continue

        # if one side exhausted, outcome is forced
        if good <= 0:
            # forced failure
            bad -= 1
        elif bad <= 0:
            # forced success
            good -= 1
            succ += 1
        else:
            # random draw without replacement
            if np.random.random() < (good / total):
                good -= 1
                succ += 1
            else:
                bad -= 1

        total -= 1
        out[m] = succ


# -----------------------------
# Core simulation kernels
# -----------------------------

@njit(cache=True)
def draw_workloads_K(
    S: int,
    N: int,
    beta_a: float,
    beta_b: float,
    mean_presence: float,
    clip_max: float,
) -> np.ndarray:
    """Draw N relative workloads from Beta(a,b), rescale so mean equals mean_presence,
    clip to clip_max, return integer shift-counts K_i per nurse."""
    rel = np.empty(N, dtype=np.float64)
    s = 0.0
    for i in range(N):
        r = beta_via_gamma(beta_a, beta_b)
        rel[i] = r
        s += r

    rel_mean = s / N
    if rel_mean <= 0.0:
        rel_mean = 1.0

    K = np.empty(N, dtype=np.int64)
    for i in range(N):
        p = mean_presence * (rel[i] / rel_mean)
        if p < 0.0:
            p = 0.0
        if p > clip_max:
            p = clip_max
        k = int(np.rint(p * S))
        if k < 0:
            k = 0
        if k > S:
            k = S
        K[i] = k

    return K


@njit(cache=True)
def draw_relative_workloads(
    N: int,
    beta_a: float,
    beta_b: float,
) -> np.ndarray:
    """Draw N relative workloads from Beta(a,b), normalized to mean=1.
    
    This captures the "shape" of a ward's workload distribution - who works
    more/less relative to the average. Can then be scaled to any mean presence.
    """
    rel = np.empty(N, dtype=np.float64)
    s = 0.0
    for i in range(N):
        r = beta_via_gamma(beta_a, beta_b)
        rel[i] = r
        s += r

    rel_mean = s / N
    if rel_mean <= 0.0:
        rel_mean = 1.0

    # Normalize so mean = 1
    for i in range(N):
        rel[i] = rel[i] / rel_mean

    return rel


@njit(cache=True)
def scale_workloads_to_presence(
    S: int,
    rel_workloads: np.ndarray,
    mean_presence: float,
    clip_max: float,
) -> np.ndarray:
    """Scale normalized relative workloads to a target mean presence.
    
    rel_workloads: array with mean=1, representing relative workload fractions
    mean_presence: target mean presence (0 to 1)
    
    Returns: integer shift counts K_i per nurse
    """
    N = rel_workloads.shape[0]
    K = np.empty(N, dtype=np.int64)
    
    for i in range(N):
        p = mean_presence * rel_workloads[i]
        if p < 0.0:
            p = 0.0
        if p > clip_max:
            p = clip_max
        k = int(np.rint(p * S))
        if k < 0:
            k = 0
        if k > S:
            k = S
        K[i] = k

    return K


@njit(cache=True)
def one_trial_metrics(
    S: int,
    K: np.ndarray,
    M: int,
    T: int,
) -> Tuple[int, int]:
    """Given roster K (shifts per nurse) and incidents M, return:
      - max overlap across nurses
      - count of nurses with overlap >= T
    """
    max_ov = 0
    c_ge_T = 0

    for i in range(K.shape[0]):
        k = int(K[i])
        ov = hypergeom_sequential(k, S - k, M)
        if ov > max_ov:
            max_ov = ov
        if ov >= T:
            c_ge_T += 1

    return max_ov, c_ge_T


# -----------------------------
# Histogram-based quantiles
# -----------------------------

@njit(cache=True)
def _clear_hist(hist: np.ndarray, upto: int) -> None:
    for k in range(upto + 1):
        hist[k] = 0


@njit(cache=True)
def _higher_quantile_from_hist(hist: np.ndarray, q: float, n: int) -> float:
    """Empirical quantile with method="higher"."""
    if n <= 0:
        return 0.0

    pos = q * (n - 1)
    idx = int(np.ceil(pos))
    if idx < 0:
        idx = 0
    elif idx > (n - 1):
        idx = n - 1

    target_rank = idx + 1
    cum = 0
    for v in range(hist.shape[0]):
        cum += int(hist[v])
        if cum >= target_rank:
            return float(v)

    return float(hist.shape[0] - 1)


# -----------------------------
# Fixed-R simulation (simple, predictable)
# -----------------------------

@njit(cache=True)
def _draw_incident_shifts(S: int, M: int) -> np.ndarray:
    """Draw M incident shift indices without replacement from [0, S-1].
    
    Uses Fisher-Yates partial shuffle for efficiency.
    """
    if M >= S:
        return np.arange(S, dtype=np.int64)
    
    # Create array of all shift indices
    pool = np.arange(S, dtype=np.int64)
    
    # Partial Fisher-Yates: only shuffle the first M elements
    for i in range(M):
        j = i + int(np.random.random() * (S - i))
        pool[i], pool[j] = pool[j], pool[i]
    
    return pool[:M].copy()


@njit(cache=True)
def one_trial_metrics_with_incidents(
    S: int,
    K: np.ndarray,
    incident_shifts: np.ndarray,
    T: int,
) -> Tuple[int, int]:
    """Given roster K and specific incident shift indices, return:
      - max overlap across nurses
      - count of nurses with overlap >= T
    
    This version takes pre-drawn incident positions rather than 
    sampling via hypergeometric, ensuring consistency across cells.
    """
    M = incident_shifts.shape[0]
    max_ov = 0
    c_ge_T = 0
    
    # For each nurse, we need to know which of their shifts overlap with incidents
    # Nurse i works shifts [0, K[i]-1] conceptually, but we need actual shift assignment
    # 
    # The original model assumes each nurse's K[i] shifts are a random sample from S.
    # For consistency, we pre-assign each nurse's working shifts, then count overlaps.
    #
    # However, the original hypergeometric model doesn't explicitly track which shifts -
    # it just computes overlap probabilistically. To maintain the same statistical model
    # while ensuring incident consistency, we:
    # 1) For each nurse, draw their K[i] working shifts (once per roster)
    # 2) Count how many of those overlap with the fixed incident_shifts
    #
    # But wait - in ensemble_rosters=False, we want the ROSTER fixed per cell.
    # The roster is the K array (how many shifts each nurse works).
    # The WHICH shifts each nurse works isn't explicitly modeled - it's implicit
    # in the hypergeometric sampling.
    #
    # For row-consistency, we need to fix incident positions and let each nurse's
    # shift assignment vary (which is what hypergeometric does implicitly).
    # But hypergeometric re-samples which shifts the nurse works each trial.
    #
    # True row-consistency requires: same incidents, same nurse shift assignments.
    # This means we need to pre-draw nurse shift assignments per (mean_presence, r).
    
    # For now, use the simpler approach: same incidents, hypergeometric overlap.
    # This still gives row-consistency for the incident component.
    
    for i in range(K.shape[0]):
        k = int(K[i])
        # Count overlap: how many of this nurse's k shifts are among incident_shifts?
        # Using hypergeometric: ngood=M incidents, nbad=S-M non-incidents, nsample=k
        ov = hypergeom_sequential(M, S - M, k)
        if ov > max_ov:
            max_ov = ov
        if ov >= T:
            c_ge_T += 1
    
    return max_ov, c_ge_T


@njit(cache=True)
def simulate_grid_fixed(
    S: int,
    N: int,
    beta_a: float,
    beta_b: float,
    incident_grid: np.ndarray,
    mean_presence_grid: np.ndarray,
    R: int,
    T: int,
    q95: float,
    q99: float,
    clip_max: float,
    ensemble_rosters: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-R Monte Carlo: exactly R trials per grid cell.

    This version is fast for:
      - ensemble_rosters=False (your greased-lightning path)
      - ensemble_rosters=True (row-wise trials, reuse each trial across all M)
    """
    I = mean_presence_grid.shape[0]
    J = incident_grid.shape[0]

    mean_max = np.zeros((I, J), dtype=np.float64)
    q95_max  = np.zeros((I, J), dtype=np.float64)
    q99_max  = np.zeros((I, J), dtype=np.float64)
    mean_c   = np.zeros((I, J), dtype=np.float64)
    prob_2plus = np.zeros((I, J), dtype=np.float64)

    # maxM (cap at S)
    maxM = 0
    for j in range(J):
        mj = int(incident_grid[j])
        if mj > maxM:
            maxM = mj
    if maxM > S:
        maxM = S
    if maxM < 0:
        maxM = 0

    # Sort columns by M so we can handle duplicates cleanly
    order = _argsort_int64(incident_grid)

    # Fill deterministic M<=0 columns immediately
    first_pos = 0
    while first_pos < J and int(incident_grid[order[first_pos]]) <= 0:
        j0 = int(order[first_pos])
        for i in range(I):
            mean_max[i, j0] = 0.0
            q95_max[i, j0]  = 0.0
            q99_max[i, j0]  = 0.0
            mean_c[i, j0]   = 0.0
            prob_2plus[i, j0] = 0.0
        first_pos += 1

    # Nothing else to do?
    if first_pos >= J or maxM == 0:
        return mean_max, q95_max, q99_max, mean_c, prob_2plus

    # -----------------------------
    # FAST ensemble_rosters=True
    # -----------------------------
    if ensemble_rosters:
        # Accumulators
        sum_max = np.zeros((I, J), dtype=np.float64)
        sum_c   = np.zeros((I, J), dtype=np.float64)
        c2      = np.zeros((I, J), dtype=np.int64)
        hist_max = np.zeros((I, J, maxM + 1), dtype=np.int64)

        # Temp buffers per trial
        overlaps = np.empty((N, maxM), dtype=np.int16)     # overlaps[n, m_idx]
        prefix   = np.empty(maxM, dtype=np.int16)          # per nurse

        for i in range(I):
            mp = float(mean_presence_grid[i])

            for r in range(R):
                # draw workloads once for this row+trial
                K = draw_workloads_K(S, N, beta_a, beta_b, mp, clip_max)

                # generate overlap prefix paths for each nurse up to maxM
                for n in range(N):
                    k = int(K[n])
                    if k < 0:
                        k = 0
                    elif k > S:
                        k = S
                    _hypergeom_prefix_counts(k, S - k, maxM, prefix)
                    for m in range(maxM):
                        overlaps[n, m] = prefix[m]

                # emit all requested M columns for this row+trial
                jpos = first_pos
                while jpos < J:
                    M = int(incident_grid[order[jpos]])
                    if M > maxM:
                        M = maxM
                    m_idx = M - 1  # M>=1 here

                    # compute max overlap and count>=T across nurses for this M
                    mx = 0
                    c_ge_T = 0
                    for n in range(N):
                        v = int(overlaps[n, m_idx])
                        if v > mx:
                            mx = v
                        if v >= T:
                            c_ge_T += 1

                    # update all columns having this same raw M (handles duplicates cleanly)
                    targetM_raw = int(incident_grid[order[jpos]])
                    while jpos < J and int(incident_grid[order[jpos]]) == targetM_raw:
                        jcol = int(order[jpos])
                        if int(incident_grid[jcol]) != int(incident_grid[order[jpos]]):
                            break  # safety; usually not needed

                        # if that column’s M differs due to cap, handle it
                        Mcol = int(incident_grid[jcol])
                        if Mcol <= 0:
                            jpos += 1
                            continue
                        if Mcol > maxM:
                            Mcol = maxM
                        if Mcol != M:
                            break

                        sum_max[i, jcol] += mx
                        sum_c[i, jcol] += c_ge_T
                        if c_ge_T >= 2:
                            c2[i, jcol] += 1
                        hist_max[i, jcol, mx] += 1
                        jpos += 1

                    # If we broke because of a mismatch, just continue outer while;
                    # this is a rare edge case if incident_grid has weird values.
                    # Normal case: we advanced through all duplicates of M.

                # end jpos loop
            # end r loop
        # end i loop

        # Finalize means + quantiles
        for j in range(J):
            Mj = int(incident_grid[j])
            if Mj <= 0:
                continue
            if Mj > maxM:
                Mj = maxM
            for i in range(I):
                mean_max[i, j] = sum_max[i, j] / R
                mean_c[i, j]   = sum_c[i, j] / R
                prob_2plus[i, j] = c2[i, j] / R
                q95_max[i, j]  = _higher_quantile_from_hist(hist_max[i, j, :], q95, R)
                q99_max[i, j]  = _higher_quantile_from_hist(hist_max[i, j, :], q99, R)

        return mean_max, q95_max, q99_max, mean_c, prob_2plus

    # -----------------------------
    # ensemble_rosters=False
    # (keep your fast path as-is)
    # -----------------------------

    # One ward "structure" across presence levels: draw relative workloads once, scale per presence
    rel_workloads = draw_relative_workloads(N, beta_a, beta_b)

    K_all = np.empty((I, N), dtype=np.int64)
    for i in range(I):
        mp = float(mean_presence_grid[i])
        K_row = scale_workloads_to_presence(S, rel_workloads, mp, clip_max)
        for n in range(N):
            K_all[i, n] = K_row[n]

    # Pre-draw a full preference ordering per nurse
    nurse_pref_orders = np.empty((N, S), dtype=np.int64)
    for n in range(N):
        perm = np.arange(S, dtype=np.int64)
        for idx in range(S - 1):
            jj = idx + int(np.random.random() * (S - idx))
            perm[idx], perm[jj] = perm[jj], perm[idx]
        for idx in range(S):
            nurse_pref_orders[n, idx] = perm[idx]

    rank = _build_rank_from_pref(nurse_pref_orders)

    sum_max = np.zeros((I, J), dtype=np.float64)
    sum_c = np.zeros((I, J), dtype=np.float64)
    c2 = np.zeros((I, J), dtype=np.int64)
    hist_max = np.zeros((I, J, maxM + 1), dtype=np.int64)

    ov = np.zeros((I, N), dtype=np.int64)

    for r in range(R):
        incidents = _draw_incident_shifts(S, maxM)

        for i in range(I):
            for n in range(N):
                ov[i, n] = 0

        jpos = first_pos
        for m in range(maxM):
            shift = int(incidents[m])

            for i in range(I):
                for n in range(N):
                    if rank[n, shift] < K_all[i, n]:
                        ov[i, n] += 1

            curM = m + 1

            while jpos < J and int(incident_grid[order[jpos]]) == curM:
                jcol = int(order[jpos])

                for i in range(I):
                    max_ov = 0
                    c_ge_T = 0
                    for n in range(N):
                        v = int(ov[i, n])
                        if v > max_ov:
                            max_ov = v
                        if v >= T:
                            c_ge_T += 1

                    sum_max[i, jcol] += max_ov
                    sum_c[i, jcol] += c_ge_T
                    if c_ge_T >= 2:
                        c2[i, jcol] += 1
                    hist_max[i, jcol, max_ov] += 1

                jpos += 1

    for j in range(J):
        M = int(incident_grid[j])
        if M <= 0:
            continue
        if M > maxM:
            M = maxM

        for i in range(I):
            mean_max[i, j] = sum_max[i, j] / R
            mean_c[i, j] = sum_c[i, j] / R
            prob_2plus[i, j] = c2[i, j] / R
            q95_max[i, j] = _higher_quantile_from_hist(hist_max[i, j, :], q95, R)
            q99_max[i, j] = _higher_quantile_from_hist(hist_max[i, j, :], q99, R)

    return mean_max, q95_max, q99_max, mean_c, prob_2plus

# -----------------------------
# Adaptive MC simulation
# -----------------------------


# -----------------------------
# Plotting utilities
# -----------------------------

def bilinear_interpolate(xg: np.ndarray, yg: np.ndarray, Z: np.ndarray, x0: float, y0: float) -> float:
    """Bilinearly interpolate Z(yg, xg) at (x0, y0)."""
    x0 = float(x0)
    y0 = float(y0)

    J = xg.shape[0]
    I = yg.shape[0]

    x0 = np.clip(x0, xg[0], xg[-1])
    y0 = np.clip(y0, yg[0], yg[-1])

    jlo = np.searchsorted(xg, x0, side='right') - 1
    jlo = max(0, min(jlo, J - 2))
    jhi = jlo + 1

    ilo = np.searchsorted(yg, y0, side='right') - 1
    ilo = max(0, min(ilo, I - 2))
    ihi = ilo + 1

    x1, x2 = xg[jlo], xg[jhi]
    y1, y2 = yg[ilo], yg[ihi]

    if x2 == x1:
        tx = 0.0
    else:
        tx = (x0 - x1) / (x2 - x1)

    if y2 == y1:
        ty = 0.0
    else:
        ty = (y0 - y1) / (y2 - y1)

    z11 = Z[ilo, jlo]
    z12 = Z[ilo, jhi]
    z21 = Z[ihi, jlo]
    z22 = Z[ihi, jhi]

    z = (1 - tx) * (1 - ty) * z11 + tx * (1 - ty) * z12 + (1 - tx) * ty * z21 + tx * ty * z22
    return z


def _mode_tag(cfg: Config) -> str:
    return "ensemble" if cfg.ensemble_rosters else "fixed"


def _save_current_dir(fig, cfg: Config, stem: str, tight: bool = True) -> None:
    if not cfg.save_figures:
        return
    mode = _mode_tag(cfg)
    fname = f"{cfg.file_prefix}_{stem}_{mode}_R{cfg.R}.{cfg.save_format}"
    path = os.path.join(os.getcwd(), fname)
    if tight:
        fig.savefig(path, dpi=cfg.save_dpi, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=cfg.save_dpi)
    print(f"Saved: {path}")


def plot_3d_surfaces(incident_grid: np.ndarray, mean_presence_grid: np.ndarray,
                     mean_max: np.ndarray, q95_max: np.ndarray, q99_max: np.ndarray,
                     cfg: Config) -> None:
    """Interactive 3D surface plot with three semi-transparent surfaces."""
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.lines import Line2D

    X, Y = np.meshgrid(incident_grid, mean_presence_grid)

    fig = plt.figure(figsize=(13, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    c_mean = "#2b83ba"
    c_q95 = "#fdae61"
    c_q99 = "#d7191c"

    ax.plot_surface(X, Y, mean_max, alpha=0.55, color=c_mean, edgecolor="none", shade=True, label="Mean")
    ax.plot_surface(X, Y, q95_max, alpha=0.45, color=c_q95, edgecolor="none", shade=True, label="Q95")
    ax.plot_surface(X, Y, q99_max, alpha=0.40, color=c_q99, edgecolor="none", shade=True, label="Q99")

    ax.set_xlabel("Number of incidents (M)", labelpad=12)
    ax.set_ylabel("Mean shift density (p̄)", labelpad=12)
    ax.set_zlabel("Max overlap", labelpad=8)

    mode_desc = "Ensemble rosters (re-drawn each trial)" if cfg.ensemble_rosters else "Fixed roster per cell"
    R_desc = f"R={cfg.R}"
    ax.set_title(f"Incident–Roster Overlap Surfaces\n({mode_desc}, {R_desc})", fontsize=12, pad=18)

    proxy_mean = Line2D([0], [0], linestyle="none", marker="s", markersize=10,
                        markerfacecolor=c_mean, alpha=0.7, label="Mean(max)")
    proxy_q95 = Line2D([0], [0], linestyle="none", marker="s", markersize=10,
                       markerfacecolor=c_q95, alpha=0.7, label="Q95(max)")
    proxy_q99 = Line2D([0], [0], linestyle="none", marker="s", markersize=10,
                       markerfacecolor=c_q99, alpha=0.7, label="Q99(max)")

    leg1 = ax.legend(handles=[proxy_mean, proxy_q95, proxy_q99], loc="upper left",
                     title="Surfaces", fontsize=10, title_fontsize=10,
                     framealpha=0.9, edgecolor="#b0b0b0")
    ax.add_artist(leg1)

    ax.view_init(elev=25, azim=-50)
    
    # Force z-axis to start from zero
    z_max = q99_max.max()
    ax.set_zlim(0, z_max * 1.05)  # small headroom above max

    # Intercept annotation
    if cfg.show_intercept:
        m0 = cfg.intercept_M
        p0 = cfg.intercept_mean_presence

        z_mean = bilinear_interpolate(incident_grid, mean_presence_grid, mean_max, m0, p0)
        z_q95 = bilinear_interpolate(incident_grid, mean_presence_grid, q95_max, m0, p0)
        z_q99 = bilinear_interpolate(incident_grid, mean_presence_grid, q99_max, m0, p0)

        ax.scatter([m0], [p0], [z_mean], s=40, color=c_mean, edgecolors="k", linewidths=0.6, depthshade=False)
        ax.scatter([m0], [p0], [z_q95], s=40, color=c_q95, edgecolors="k", linewidths=0.6, depthshade=False)
        ax.scatter([m0], [p0], [z_q99], s=40, color=c_q99, edgecolors="k", linewidths=0.6, depthshade=False)

        z_floor = ax.get_zlim()[0]
        ax.scatter([m0], [p0], [z_floor], marker="x", s=55, color="k", linewidths=2, depthshade=False)
        ax.plot([m0, m0], [p0, p0], [z_floor, z_q99], color="k", linestyle="--", linewidth=1.0, alpha=0.6)

        intercept_handles = [
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=c_mean, markeredgecolor="k", markersize=7,
                   label=f"Mean(max) ≈ {z_mean:.2f}"),
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=c_q95, markeredgecolor="k", markersize=7,
                   label=f"Q95(max) ≈ {z_q95:.2f}"),
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=c_q99, markeredgecolor="k", markersize=7,
                   label=f"Q99(max) ≈ {z_q99:.2f}"),
        ]

        leg2 = fig.legend(
            handles=intercept_handles,
            loc="lower left",
            bbox_to_anchor=(0.17, 0.08),
            frameon=True,
            title=cfg.intercept_label,
            fontsize=10,
            title_fontsize=10,
            handlelength=0.8,
            labelspacing=0.35,
            borderpad=0.6,
        )
        frame2 = leg2.get_frame()
        frame2.set_facecolor("#f2f2f2")
        frame2.set_alpha(0.91)
        frame2.set_edgecolor("#b0b0b0")
        frame2.set_linewidth(0.9)

    _save_current_dir(fig, cfg, "3d_surfaces", tight=False)
    plt.show()


def heatmap(data: np.ndarray, incident_grid: np.ndarray, mean_presence_grid: np.ndarray,
            title: str, cbar_label: str, cfg: Config, stem: str) -> None:
    """2D heatmap."""
    fig = plt.figure(figsize=(9.5, 5.0))
    ax = fig.add_subplot(111)

    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        interpolation=cfg.heatmap_interpolation,
        extent=[incident_grid.min(), incident_grid.max(), mean_presence_grid.min(), mean_presence_grid.max()],
    )
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    ax.set_xlabel("Number of incidents (M)")
    ax.set_ylabel("Mean shift density (p̄)")

    if cfg.show_intercept and cfg.annotate_intercept_on_heatmaps:
        ax.scatter([cfg.intercept_M], [cfg.intercept_mean_presence], marker="x", s=80, color="k", linewidths=2)
        ax.text(cfg.intercept_M, cfg.intercept_mean_presence, "  " + cfg.intercept_label,
                fontsize=9, va="center", ha="left", color="k")

    plt.tight_layout()
    _save_current_dir(fig, cfg, stem)
    plt.show()


# -----------------------------
# Manifold export (JSON)
# -----------------------------

def export_manifold_json(
    incident_grid: np.ndarray,
    mean_presence_grid: np.ndarray,
    mean_max: np.ndarray,
    q95_max: np.ndarray,
    q99_max: np.ndarray,
    cfg: Config,
) -> str:
    """Export the manifold pack for a lightweight JS viewer."""
    if len(incident_grid) < 50:
        print("⚠ Warning: Exporting with coarse incident grid. For full detail, set export=True before running.")

    out_dir = os.getcwd()
    mode = _mode_tag(cfg)
    fname = f"{cfg.export_json_stem}_{mode}_R{cfg.R}.json"
    path = os.path.join(out_dir, fname)

    pack = {
        "version": "1.3",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": mode,
        "params": {
            "S": cfg.S,
            "N": cfg.N,
            "beta_a": cfg.beta_a,
            "beta_b": cfg.beta_b,
            "ensemble_rosters": cfg.ensemble_rosters,
            "presence_clip_max": cfg.presence_clip_max,
            "R": cfg.R,
            "q95": cfg.q95,
            "q99": cfg.q99,
            "seed": cfg.seed,
        },
        "axes": {
            "incident_grid": [int(x) for x in incident_grid],
            "mean_presence_grid": [round(float(x), 4) for x in mean_presence_grid],
            "shape": [int(mean_presence_grid.shape[0]), int(incident_grid.shape[0])],
        },
        "surfaces": {
            "mean_max": [round(float(x), 3) for x in mean_max.ravel(order="C")],
            "q95_max": [int(x) for x in q95_max.ravel(order="C")],
            "q99_max": [int(x) for x in q99_max.ravel(order="C")],
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        if cfg.export_pretty:
            json.dump(pack, f, ensure_ascii=False, indent=2)
        else:
            json.dump(pack, f, ensure_ascii=False, separators=(",", ":"))

    print(f"Exported manifold JSON: {path}")
    
    # Also save a fixed-name copy for easy HTML viewer integration
    fixed_name = "manifold_roster_pack.json"
    fixed_path = os.path.join(out_dir, fixed_name)
    with open(fixed_path, "w", encoding="utf-8") as f:
        if cfg.export_pretty:
            json.dump(pack, f, ensure_ascii=False, indent=2)
        else:
            json.dump(pack, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Exported fixed-name copy: {fixed_path}")
    
    return path


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    cfg = Config()

    np.random.seed(cfg.seed)

    incident_grid = np.array(cfg.get_incident_grid(), dtype=np.int64)
    mean_presence_grid = np.array(cfg.mean_presence_grid, dtype=np.float64)

    total_cells = len(mean_presence_grid) * len(incident_grid)
    print(f"Grid: {len(mean_presence_grid)} × {len(incident_grid)} = {total_cells} cells")
    print(f"Rosters: {'Ensemble (new each trial)' if cfg.ensemble_rosters else 'Fixed per cell'}")
    print(f"Trials: R={cfg.R}")
    print()

    t0 = time.time()

    mean_max, q95_max, q99_max, mean_c, prob_2plus = simulate_grid_fixed(
        S=cfg.S,
        N=cfg.N,
        beta_a=cfg.beta_a,
        beta_b=cfg.beta_b,
        incident_grid=incident_grid,
        mean_presence_grid=mean_presence_grid,
        R=cfg.R,
        T=cfg.threshold_T,
        q95=cfg.q95,
        q99=cfg.q99,
        clip_max=cfg.presence_clip_max,
        ensemble_rosters=cfg.ensemble_rosters,
    )

    dt = time.time() - t0
    print(f"Done. Time: {dt:.2f}s ({dt/total_cells*1000:.1f}ms/cell)")

    # Export
    if cfg.export:
        export_manifold_json(incident_grid, mean_presence_grid, mean_max, q95_max, q99_max, cfg)

    # Intercept readout
    if cfg.show_intercept:
        m0 = cfg.intercept_M
        grid_values = cfg.get_incident_grid()

        if not cfg.export and m0 not in grid_values:
            print(f"⚠ Note: Intercept M={m0} requires interpolation (coarse grid)")

        z_mean = bilinear_interpolate(incident_grid, mean_presence_grid, mean_max,
                                      cfg.intercept_M, cfg.intercept_mean_presence)
        z_q95 = bilinear_interpolate(incident_grid, mean_presence_grid, q95_max,
                                      cfg.intercept_M, cfg.intercept_mean_presence)
        z_q99 = bilinear_interpolate(incident_grid, mean_presence_grid, q99_max,
                                      cfg.intercept_M, cfg.intercept_mean_presence)
        print(
            f"Intercept @ M={cfg.intercept_M:g}, p̄={cfg.intercept_mean_presence:g} -> "
            f"mean(max)={z_mean:.3f}, Q95(max)={z_q95:.3f}, Q99(max)={z_q99:.3f}"
        )

    # Plots
    if cfg.show_3d:
        plot_3d_surfaces(incident_grid, mean_presence_grid, mean_max, q95_max, q99_max, cfg)

    if cfg.show_heatmaps:
        heatmap(mean_max, incident_grid, mean_presence_grid,
                title=f"Mean of max overlap across nurses (R={cfg.R})",
                cbar_label="E[max overlap]", cfg=cfg, stem="heat_mean_max")
        heatmap(q95_max, incident_grid, mean_presence_grid,
                title=f"Q95 of max overlap across nurses (R={cfg.R})",
                cbar_label="Q95[max overlap]", cfg=cfg, stem="heat_q95_max")
        heatmap(q99_max, incident_grid, mean_presence_grid,
                title=f"Q99 of max overlap across nurses (R={cfg.R})",
                cbar_label="Q99[max overlap]", cfg=cfg, stem="heat_q99_max")
        heatmap(mean_c, incident_grid, mean_presence_grid,
                title=f"Mean # nurses with overlap ≥ T (T={cfg.threshold_T}, R={cfg.R})",
                cbar_label="E[# nurses ≥ T]", cfg=cfg, stem="heat_mean_count_ge_T")
        heatmap(prob_2plus, incident_grid, mean_presence_grid,
                title=f"P(≥2 nurses with overlap ≥ T) (T={cfg.threshold_T}, R={cfg.R})",
                cbar_label="Probability", cfg=cfg, stem="heat_prob_2plus_ge_T")


if __name__ == "__main__":
    main()