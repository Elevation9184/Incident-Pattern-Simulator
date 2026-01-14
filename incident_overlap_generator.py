"""Incident–Roster Dynamics: Manifold / Surface Visualisation (Numba-optimised)

Generates 3D semi-transparent surfaces and 2D heatmaps showing how the
maximum incident–roster overlap (across nurses) scales with:
  - number of incidents (M)
  - mean shift density / presence fraction (p̄)

It also optionally computes “multi-suspect” metrics:
  - E[# nurses with overlap >= T]
  - P(# nurses with overlap >= T >= 2)

Two modelling modes:
  1) ensemble_rosters=True: re-draw workloads each trial ("high-workload role" rotates)
  2) ensemble_rosters=False: fix workloads per grid cell, re-sample incidents only

This is designed so you can crank MC repeats (R) up and see surfaces stabilise.

Dependencies:
  - numpy
  - matplotlib
  - numba

Run:
  python shift_surface_analysis.py

Notes:
  - We implement Beta(a,b) via Gamma draws (Numba-friendly).
  - We implement Hypergeometric via sequential without-replacement draws (exact).
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

    # Monte Carlo per grid cell
    R: int = 5000  # increase (e.g., 200, 500, 1000) to stabilise surfaces

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
    show_heatmaps: bool = False

    # Intercept annotation (e.g., Letby-style "suspicious incidents" + estimated mean presence)
    # Used to place a marker on the 3D surface plot and (optionally) on heatmaps.
    show_intercept: bool = True
    intercept_M: float = 61.0
    intercept_mean_presence: float = 0.22
    # Tip: include an "x" here so the label matches the plotted x-marker at the base.
    intercept_label: str = "Intercept x (M=61, p̄=0.22)"
    intercept_label: str = f"Intercept $\\mathbf{{x}}$ (M={int(intercept_M)}, p̄={intercept_mean_presence:.2f})"
    annotate_intercept_on_heatmaps: bool = False

    # Saving
    save_figures: bool = True          # save figures to current working directory
    save_format: str = "png"            # png, pdf, svg, ...
    save_dpi: int = 200                 # only relevant for raster formats  
    file_prefix: str = "manifold"       # prefix for saved filenames

    # Data export (for JS viewer / later reuse)
    export: bool = True
    export_json_stem: str = "manifold_pack"
    export_pretty: bool = False  # pretty JSON (larger, slower) if True

    # Heatmap rendering
    # - "nearest" = crisp rectangles (default, shows grid explicitly)
    # - "bilinear"/"bicubic" = blended/continuous appearance (presentation-friendly)
    heatmap_interpolation: str = "nearest"

    def get_incident_grid(self) -> Tuple[int, ...]:
        """Return incident grid based on export setting.

        When export=True: Full unit-level granularity (1, 2, 3, ..., 80) - 80 points
        When export=False: Coarse granularity (10, 20, 30, ..., 80) - 8 points for ~10x speedup
        """
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
    """Exact hypergeometric draw via sequential without-replacement sampling.

    Returns number of "good" draws when sampling nsample from a population with
    ngood good and nbad bad.

    Complexity: O(nsample). Fast enough for typical M<=100.
    """
    good = ngood
    bad = nbad
    total = good + bad
    succ = 0

    # cap if nsample exceeds total
    if nsample > total:
        nsample = total

    for _ in range(nsample):
        if total <= 0:
            break
        # probability of drawing a "good" item
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


@njit(cache=True)
def simulate_grid(
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
    """Main driver.

    Returns arrays with shape (len(mean_presence_grid), len(incident_grid)):
      mean_max, q95_max, q99_max, mean_count_ge_T, prob_2plus_ge_T

    q95_max and q99_max are quantiles over the R trial maxima.
    """

    I = mean_presence_grid.shape[0]
    J = incident_grid.shape[0]

    mean_max = np.zeros((I, J), dtype=np.float64)
    q95_max = np.zeros((I, J), dtype=np.float64)
    q99_max = np.zeros((I, J), dtype=np.float64)

    mean_c = np.zeros((I, J), dtype=np.float64)
    prob_2plus = np.zeros((I, J), dtype=np.float64)

    # temp buffers
    maxima = np.empty(R, dtype=np.int64)
    counts = np.empty(R, dtype=np.int64)

    for i in range(I):
        mp = float(mean_presence_grid[i])
        for j in range(J):
            M = int(incident_grid[j])

            # fixed roster per cell (if chosen)
            K_fixed = draw_workloads_K(S, N, beta_a, beta_b, mp, clip_max)

            for r in range(R):
                if ensemble_rosters:
                    K = draw_workloads_K(S, N, beta_a, beta_b, mp, clip_max)
                else:
                    K = K_fixed

                mx, c = one_trial_metrics(S, K, M, T)
                maxima[r] = mx
                counts[r] = c

            # aggregate
            mean_max[i, j] = maxima.mean()

            # quantiles: sort maxima (R is modest)
            tmp = np.sort(maxima.copy())

            def _higher_quantile(q: float) -> float:
                # Mimics NumPy quantile(..., method="higher") for 1D data:
                # take the element at ceil(q*(R-1)).
                pos = q * (R - 1)
                idx = int(np.ceil(pos))
                if idx < 0:
                    idx = 0
                elif idx > (R - 1):
                    idx = R - 1
                return float(tmp[idx])

            q95_max[i, j] = _higher_quantile(q95)
            q99_max[i, j] = _higher_quantile(q99)

            mean_c[i, j] = counts.mean()

            # probability of >= 2 nurses crossing threshold
            c2 = 0
            for r in range(R):
                if counts[r] >= 2:
                    c2 += 1
            prob_2plus[i, j] = c2 / R

    return mean_max, q95_max, q99_max, mean_c, prob_2plus


# -----------------------------
# Plotting
# -----------------------------

def bilinear_interpolate(xg: np.ndarray, yg: np.ndarray, Z: np.ndarray, x0: float, y0: float) -> float:
    """Bilinearly interpolate Z(yg, xg) at (x0, y0).

    xg: shape (J,), increasing
    yg: shape (I,), increasing
    Z : shape (I, J) where rows correspond to yg, cols correspond to xg

    If (x0, y0) lies outside the grid, values are clamped to the nearest edge.
    """
    x0 = float(x0)
    y0 = float(y0)

    J = xg.shape[0]
    I = yg.shape[0]

    # --- x indices ---
    if x0 <= xg[0]:
        j0 = j1 = 0
        wx = 0.0
    elif x0 >= xg[J - 1]:
        j0 = j1 = J - 1
        wx = 0.0
    else:
        j1 = int(np.searchsorted(xg, x0))
        j0 = j1 - 1
        dx = float(xg[j1] - xg[j0])
        wx = 0.0 if dx == 0 else (x0 - float(xg[j0])) / dx

    # --- y indices ---
    if y0 <= yg[0]:
        i0 = i1 = 0
        wy = 0.0
    elif y0 >= yg[I - 1]:
        i0 = i1 = I - 1
        wy = 0.0
    else:
        i1 = int(np.searchsorted(yg, y0))
        i0 = i1 - 1
        dy = float(yg[i1] - yg[i0])
        wy = 0.0 if dy == 0 else (y0 - float(yg[i0])) / dy

    z00 = float(Z[i0, j0])
    z01 = float(Z[i0, j1])
    z10 = float(Z[i1, j0])
    z11 = float(Z[i1, j1])

    z0 = (1.0 - wx) * z00 + wx * z01
    z1 = (1.0 - wx) * z10 + wx * z11
    return (1.0 - wy) * z0 + wy * z1



def _mode_tag(cfg: Config) -> str:
    return "ensemble_rosters" if cfg.ensemble_rosters else "fixed_roster"


def _save_current_dir(fig, cfg: Config, stem: str, *, tight: bool = True) -> None:
    """Save figure to current working directory if enabled.

    NOTE: For the 3D surface plot we often *don't* want bbox_inches='tight'
    because it collapses the extra canvas we've intentionally added for legends.
    """
    if not cfg.save_figures:
        return
    out_dir = os.getcwd()
    fname = f"{cfg.file_prefix}_{stem}_R{cfg.R}_{_mode_tag(cfg)}.{cfg.save_format}"
    path = os.path.join(out_dir, fname)

    save_kwargs = {"bbox_inches": "tight"} if tight else {}
    # DPI only meaningful for raster formats
    if cfg.save_format.lower() in ("png", "jpg", "jpeg", "tif", "tiff", "webp"):
        save_kwargs["dpi"] = cfg.save_dpi

    fig.savefig(path, **save_kwargs)
    print(f"Saved: {path}")

def plot_3d_surfaces(incident_grid: np.ndarray, mean_presence_grid: np.ndarray,
                     mean_max: np.ndarray, q95_max: np.ndarray, q99_max: np.ndarray,
                     cfg: Config) -> None:
    """3D plot with a clear colour key.

    Matplotlib's 3D surfaces don't automatically create a legend, so we add
    proxy colour patches that match the surface colours.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    X, Y = np.meshgrid(incident_grid, mean_presence_grid)

    # Slightly larger canvas so figure-level legends have room without
    # shrinking or occluding the plotted surfaces.
    fig = plt.figure(figsize=(12.5, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    # Keep the plot itself roughly the same size by reserving margins.
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.90)   # top=0.92

    # Explicit colours so the key is unambiguous.
    c_mean = "tab:blue"
    c_q95  = "tab:orange"
    c_q99  = "tab:green"

    ax.plot_surface(X, Y, mean_max,  color=c_mean, alpha=0.55, linewidth=0, antialiased=True)
    ax.plot_surface(X, Y, q95_max,   color=c_q95,  alpha=0.35, linewidth=0, antialiased=True)
    ax.plot_surface(X, Y, q99_max,   color=c_q99,  alpha=0.20, linewidth=0, antialiased=True)

    mode = _mode_tag(cfg)

    title = "Incident–roster overlap surfaces under neutrality"
    subtitle = f"({cfg.N} nurses, {cfg.S} shifts; observations over {cfg.R:,} simulated rosters)"

    fig.suptitle(title, fontsize=16, fontweight="semibold", y=0.97)
    fig.text(0.5, 0.935, subtitle, ha="center", va="top", fontsize=11, alpha=0.9)

    ax.set_title(
        f"Incident–roster overlap manifold (N={cfg.N}, S={cfg.S}, R={cfg.R}, mode={mode})\n"
        f"Surfaces: mean(max overlap), Q95(max overlap), Q99(max overlap)"
    )
    ax.set_title("")

    ax.set_xlabel("Number of incidents (M)")
    ax.set_ylabel("Mean shift density (p̄)")
    ax.set_zlabel("Max overlaps (any nurse, per run)")

    # ---- Upper legend (unchanged) ----
    legend_elements = [
        Patch(facecolor=c_mean, edgecolor="k", alpha=0.55, label="Mean of max overlap"),
        Patch(facecolor=c_q95,    edgecolor="k", alpha=0.35, label="Q95 of max"),
        Patch(facecolor=c_q99,  edgecolor="k", alpha=0.20, label="Q99 of max"),
    ]
    leg = fig.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.22, 0.90),
        frameon=True,
    )
    frame = leg.get_frame()
    frame.set_facecolor("#f2f2f2")
    frame.set_alpha(0.92)
    frame.set_edgecolor("#b0b0b0")
    frame.set_linewidth(0.9)

    # Pin the z-floor at zero so the "base" plane isn't visually below z=0.
    z_top = float(np.nanmax(q99_max))
    ax.set_zlim(0.0, max(1.0, z_top) * 1.05)

    # --- Optional intercept marker (e.g., M=61, p̄=0.22) ---
    if cfg.show_intercept:
        m0 = float(cfg.intercept_M)
        p0 = float(cfg.intercept_mean_presence)

        z_mean = bilinear_interpolate(incident_grid, mean_presence_grid, mean_max, m0, p0)
        z_q95 = bilinear_interpolate(incident_grid, mean_presence_grid, q95_max, m0, p0)
        z_q99 = bilinear_interpolate(incident_grid, mean_presence_grid, q99_max, m0, p0)

        # Mark the three surface intercepts (make them a bit "poppier" than the translucent surfaces)
        marker_s = 70
        marker_lw = 0.9
        ax.scatter([m0], [p0], [z_mean], color=c_mean, s=marker_s, depthshade=False,
                   edgecolors="k", linewidths=marker_lw, alpha=1.0)
        ax.scatter([m0], [p0], [z_q95], color=c_q95, s=marker_s, depthshade=False,
                   edgecolors="k", linewidths=marker_lw, alpha=1.0)
        ax.scatter([m0], [p0], [z_q99], color=c_q99, s=marker_s, depthshade=False,
                   edgecolors="k", linewidths=marker_lw, alpha=1.0)

        # A base marker + vertical guide line up to the Q99 intercept (helps the eye)
        z_floor = ax.get_zlim()[0]
        ax.scatter([m0], [p0], [z_floor], marker="x", s=55, color="k", linewidths=2, depthshade=False)
        ax.plot([m0, m0], [p0, p0], [z_floor, z_q99], color="k", linestyle="--", linewidth=1.0, alpha=0.6)

        # ---- Lower legend (NEW): intercept readout with coloured marker prefixes ----
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

    # Don't use tight bbox for this plot: we want the extra canvas to remain.
    _save_current_dir(fig, cfg, "3d_surfaces", tight=False)
    plt.show()


def heatmap(data: np.ndarray, incident_grid: np.ndarray, mean_presence_grid: np.ndarray,
            title: str, cbar_label: str, cfg: Config, stem: str) -> None:
    """2D heatmap.

    cfg.heatmap_interpolation controls the look:
      - nearest  : crisp grid rectangles (default)
      - bilinear : blended / continuous appearance
      - bicubic  : even smoother (can over-smooth small grids)
    """
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

    # Optional intercept marker
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
    """Export the manifold pack for a lightweight JS viewer.

    Arrays are exported row-major (i index = mean_presence_grid, j index = incident_grid).
    """
    # Validate grid granularity
    if len(incident_grid) < 50:  # Heuristic: coarse grid detection
        print("⚠ Warning: Exporting with coarse incident grid. For full detail, set export=True before running.")

    out_dir = os.getcwd()
    mode = _mode_tag(cfg)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{cfg.export_json_stem}_{mode}_R{cfg.R}_{ts}.json"
    path = os.path.join(out_dir, fname)

    pack = {
        "version": "1.1",
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
    return path

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    cfg = Config()

    np.random.seed(cfg.seed)  # sets Numba RNG seed too

    incident_grid = np.array(cfg.get_incident_grid(), dtype=np.int64)
    mean_presence_grid = np.array(cfg.mean_presence_grid, dtype=np.float64)

    t0 = time.time()

    mean_max, q95_max, q99_max, mean_c, prob_2plus = simulate_grid(
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
    print(f"Done. Grid: {len(mean_presence_grid)}x{len(incident_grid)} | R={cfg.R} | time={dt:.2f}s")

    # --- Optional export for JS viewer ---
    if cfg.export:
        export_manifold_json(incident_grid, mean_presence_grid, mean_max, q95_max, q99_max, cfg)

    # --- Intercept readout (interpolated from the manifold grids) ---
    if cfg.show_intercept:
        m0 = cfg.intercept_M
        grid_values = cfg.get_incident_grid()

        # Warn if intercept requires interpolation in coarse mode
        if not cfg.export and m0 not in grid_values:
            print(f"⚠ Note: Intercept M={m0} requires interpolation (export=False, using coarse grid: {list(grid_values)})")

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

    # --- Plots ---
    if cfg.show_3d:
        plot_3d_surfaces(incident_grid, mean_presence_grid, mean_max, q95_max, q99_max, cfg)

    if cfg.show_heatmaps:
        heatmap(
            mean_max,
            incident_grid,
            mean_presence_grid,
            title=f"Mean of max overlap across nurses (R={cfg.R})",
            cbar_label="E[max overlap]",
            cfg=cfg,
            stem="heat_mean_max",
        )
        heatmap(
            q95_max,
            incident_grid,
            mean_presence_grid,
            title=f"Q95 of max overlap across nurses (R={cfg.R})",
            cbar_label="Q95[max overlap]",
            cfg=cfg,
            stem="heat_q95_max",
        )
        heatmap(
            q99_max,
            incident_grid,
            mean_presence_grid,
            title=f"Q99 of max overlap across nurses (R={cfg.R})",
            cbar_label="Q99[max overlap]",
            cfg=cfg,
            stem="heat_q99_max",
        )
        heatmap(
            mean_c,
            incident_grid,
            mean_presence_grid,
            title=f"Mean # nurses with overlap ≥ T (T={cfg.threshold_T}, R={cfg.R})",
            cbar_label="E[# nurses ≥ T]",
            cfg=cfg,
            stem="heat_mean_count_ge_T",
        )
        heatmap(
            prob_2plus,
            incident_grid,
            mean_presence_grid,
            title=f"P(≥2 nurses with overlap ≥ T) (T={cfg.threshold_T}, R={cfg.R})",
            cbar_label="Probability",
            cfg=cfg,
            stem="heat_prob_2plus_ge_T",
        )


if __name__ == "__main__":
    main()
