"""
FFS (Forward Flux Sampling) Data Visualization
================================================
Visualizes FFS simulation results. Handles two data formats:

  - FFS format: columns like ic_time, flux_rate_per_day, lambda0_P_forward, ...
  - IFS format: columns like init_time, n_reached_B, prob_lambda0, rate_lambda0_per_day, ...

Requirements:
    pip install matplotlib numpy pandas

Usage:
    python ffs_visualize.py --ffs data_ffs.tsv --ifs data_ifs.tsv
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter

# ── Style setup ──────────────────────────────────────────────────────────────

DARK_STYLE = {
    "figure.facecolor": "#0a0e17",
    "axes.facecolor": "#111827",
    "axes.edgecolor": "#1e2d3d",
    "axes.labelcolor": "#94a3b8",
    "text.color": "#e2e8f0",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "grid.color": "#1e2d3d",
    "grid.alpha": 0.6,
    "font.family": "monospace",
    "font.size": 9,
}

COLORS = {
    "accent": "#38bdf8",
    "purple": "#a78bfa",
    "green": "#34d399",
    "orange": "#fb923c",
    "pink": "#f472b6",
    "red": "#ef4444",
    "dim": "#64748b",
}

LAMBDA_COLORS = [
    COLORS["accent"],
    COLORS["purple"],
    COLORS["green"],
    COLORS["orange"],
    COLORS["pink"],
]


def apply_style():
    """Apply the dark plotting style."""
    plt.rcParams.update(DARK_STYLE)


# ── Data loading & normalization ─────────────────────────────────────────────


def detect_format(df: pd.DataFrame) -> str:
    """Detect whether a DataFrame is 'ffs' or 'ifs' format."""
    cols = set(df.columns)
    if "ic_time" in cols and "ffs_speedup" in cols:
        return "ffs"
    if "init_time" in cols and "prob_lambda0" in cols:
        return "ifs"
    if "ic_time" in cols:
        return "ffs"
    if "init_time" in cols:
        return "ifs"
    raise ValueError(
        f"Cannot detect data format. Expected 'ic_time' (FFS) or 'init_time' (IFS) column. "
        f"Got columns: {sorted(cols)[:15]}..."
    )


def load_data(path: str) -> pd.DataFrame:
    """Load a tab-separated FFS/IFS results file and normalize columns."""
    df = pd.read_csv(path)
    return df


def get_time_col(df: pd.DataFrame) -> str:
    """Return the time column name for this dataframe."""
    fmt = df.attrs.get("format", detect_format(df))
    return "ic_time" if fmt == "ffs" else "init_time"


def short_labels(df: pd.DataFrame) -> list[str]:
    """Create short x-axis labels, auto-detecting the time column."""
    time_col = get_time_col(df)
    times = pd.to_datetime(df[time_col])
    return [t.strftime("%b %d\n%HZ") for t in times]


def get_n_interfaces(df: pd.DataFrame) -> int:
    """Detect how many lambda interfaces exist in the data."""
    fmt = df.attrs.get("format", detect_format(df))
    if fmt == "ffs":
        return sum(1 for c in df.columns if c.startswith("lambda") and c.endswith("_P_forward"))
    else:
        return sum(1 for c in df.columns if c.startswith("prob_lambda"))


def get_pforward_col(df: pd.DataFrame, i: int) -> str:
    """Return the P_forward column name for interface i."""
    fmt = df.attrs.get("format", detect_format(df))
    if fmt == "ffs":
        return f"lambda{i}_P_forward"
    else:
        return f"prob_lambda{i}"


def get_pforward(df: pd.DataFrame, i: int) -> pd.Series:
    """Get P_forward values for interface i."""
    col = get_pforward_col(df, i)
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)


def get_rate_col(df: pd.DataFrame, i: int) -> Optional[str]:
    """Return the rate column for interface i (IFS format)."""
    col = f"rate_lambda{i}_per_day"
    return col if col in df.columns else None


def get_outcome_data(df: pd.DataFrame, i: int) -> Optional[dict]:
    """
    Get success/failure/extratropical counts for interface i.
    Returns None if data not available (e.g. IFS format).
    """
    fmt = df.attrs.get("format", detect_format(df))
    if fmt == "ffs":
        s_col = f"lambda{i}_successes"
        f_col = f"lambda{i}_failures"
        e_col = f"lambda{i}_extratropical"
        if all(c in df.columns for c in [s_col, f_col, e_col]):
            return {
                "success": df[s_col].mean(),
                "failure": df[f_col].mean(),
                "extratropical": df[e_col].mean(),
            }
    return None


# ── Plotting functions ───────────────────────────────────────────────────────


def _no_data(ax, msg="Data not available for this format"):
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, color=COLORS["dim"], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_rate_evolution(ax, df, xlabels=None):
    """Flux rate, FFS rate, direct formation rate over IC windows."""
    if xlabels is None:
        xlabels = short_labels(df)
    x = np.arange(len(df))
    fmt = df.attrs.get("format", detect_format(df))

    if fmt == "ffs":
        ax.plot(x, df["flux_rate_per_day"], color=COLORS["accent"], marker="o", ms=4, lw=1.8, label="Flux rate")
        ax.set_ylabel("Flux rate (day⁻¹)", color=COLORS["accent"])
        ax.tick_params(axis="y", colors=COLORS["accent"])

        ax2 = ax.twinx()
        ax2.plot(x, df["ffs_rate_per_day"], color=COLORS["purple"], marker="s", ms=3, lw=1.5, label="FFS rate")
        ax2.plot(x, df["direct_formation_rate_per_day"], color=COLORS["green"], marker="^", ms=3, lw=1.5, label="Direct rate")
        ax2.set_ylabel("FFS / Direct rate (day⁻¹)", color=COLORS["purple"])
        ax2.tick_params(axis="y", colors=COLORS["purple"])
        ax2.spines["right"].set_color(COLORS["purple"])

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7, framealpha=0.3)

    else:  # IFS
        n_iface = get_n_interfaces(df)
        has_any = False
        for i in range(n_iface):
            rate_col = get_rate_col(df, i)
            if rate_col and rate_col in df.columns:
                vals = df[rate_col]
                if vals.notna().any() and (vals > 0).any():
                    ax.plot(x, vals, color=LAMBDA_COLORS[i % len(LAMBDA_COLORS)],
                            marker="o", ms=3, lw=1.5, label=f"λ{i} rate")
                    has_any = True
        if not has_any:
            _no_data(ax, "No nonzero rate columns found")
            return
        ax.set_ylabel("Rate (day⁻¹)")
        ax.legend(fontsize=7, framealpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6)
    ax.set_title("Rate Evolution", fontsize=10, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3)


def plot_speedup(ax, df, xlabels=None):
    """FFS speedup bar chart (FFS format only)."""
    if xlabels is None:
        xlabels = short_labels(df)
    x = np.arange(len(df))

    if "ffs_speedup" not in df.columns:
        _no_data(ax, "ffs_speedup column not in data")
        ax.set_title("FFS Speedup", fontsize=10, fontweight="bold", pad=10)
        return

    speeds = df["ffs_speedup"].values
    colors = [COLORS["accent"] if s > 30 else COLORS["purple"] if s > 20 else COLORS["dim"] for s in speeds]
    ax.bar(x, speeds, color=colors, alpha=0.8, edgecolor="none", width=0.7)
    ax.axhline(y=np.mean(speeds), color=COLORS["orange"], ls="--", lw=1, alpha=0.7,
               label=f"Mean = {np.mean(speeds):.1f}×")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6)
    ax.set_ylabel("Speedup factor")
    ax.set_title("FFS Speedup vs Brute Force", fontsize=10, fontweight="bold", pad=10)
    ax.legend(fontsize=7, framealpha=0.3)
    ax.grid(True, axis="y", alpha=0.3)


def plot_pforward(ax, df, xlabels=None):
    """P_forward for each lambda interface (works for both formats)."""
    if xlabels is None:
        xlabels = short_labels(df)
    x = np.arange(len(df))
    n_iface = get_n_interfaces(df)

    plotted = False
    for i in range(n_iface):
        vals = get_pforward(df, i)
        if vals.notna().any():
            ax.plot(x, vals, color=LAMBDA_COLORS[i % len(LAMBDA_COLORS)],
                    marker="o", ms=3, lw=1.5, label=f"λ{i}")
            plotted = True

    if not plotted:
        _no_data(ax)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6)
    ax.set_ylabel("P_forward")
    ax.set_title("P_forward by λ-Interface", fontsize=10, fontweight="bold", pad=10)
    ax.legend(fontsize=7, ncol=min(n_iface, 6), loc="upper center", framealpha=0.3)
    ax.grid(True, alpha=0.3)


def plot_trajectories(ax, df, xlabels=None):
    """Trajectories and direct B formations (FFS) or n_reached_B (IFS)."""
    if xlabels is None:
        xlabels = short_labels(df)
    x = np.arange(len(df))
    fmt = df.attrs.get("format", detect_format(df))

    if fmt == "ffs":
        ax.bar(x - 0.15, df["flux_trajectories"], width=0.3, color=COLORS["accent"], alpha=0.6, label="Trajectories")
        ax.set_ylabel("Trajectories", color=COLORS["accent"])
        ax2 = ax.twinx()
        ax2.bar(x + 0.15, df["flux_direct_B_formations"], width=0.3, color=COLORS["red"], alpha=0.7, label="Direct B formations")
        ax2.set_ylabel("Direct B formations", color=COLORS["red"])
        ax2.tick_params(axis="y", colors=COLORS["red"])
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7, framealpha=0.3)
    else:
        ifs_cols = [
            ("n_starts_in_A", "Starts in A", COLORS["accent"]),
            ("n_reached_B", "Reached B", COLORS["green"]),
            ("n_B_from_A", "B from A (direct)", COLORS["red"]),
        ]
        available = [(col, label, color) for col, label, color in ifs_cols if col in df.columns and df[col].notna().any()]
        n = len(available)
        if n == 0:
            _no_data(ax, "No trajectory/count columns found")
        else:
            w = 0.7 / n
            for idx, (col, label, color) in enumerate(available):
                offset = (idx - (n - 1) / 2) * w
                ax.bar(x + offset, df[col], width=w, color=color, alpha=0.7, label=label)
        ax.set_ylabel("Count")
        ax.legend(fontsize=7, framealpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6)
    ax.set_title("Trajectories & Transitions", fontsize=10, fontweight="bold", pad=10)
    ax.grid(True, axis="y", alpha=0.3)


def plot_cost_comparison(ax, df, xlabels=None):
    """FFS vs brute force cost per event (log scale). FFS format only."""
    if xlabels is None:
        xlabels = short_labels(df)
    x = np.arange(len(df))

    has_ffs_cost = "cost_per_event_days" in df.columns
    has_bf_cost = "bf_days_per_event" in df.columns

    if not has_ffs_cost and not has_bf_cost:
        _no_data(ax, "Cost columns not in data")
        ax.set_title("Computational Cost", fontsize=10, fontweight="bold", pad=10)
        return

    if has_ffs_cost:
        ax.fill_between(x, df["cost_per_event_days"], alpha=0.2, color=COLORS["accent"])
        ax.plot(x, df["cost_per_event_days"], color=COLORS["accent"], marker="o", ms=4, lw=1.8, label="FFS cost/event")
    if has_bf_cost:
        ax.fill_between(x, df["bf_days_per_event"], alpha=0.1, color=COLORS["red"])
        ax.plot(x, df["bf_days_per_event"], color=COLORS["red"], marker="s", ms=4, lw=1.8, label="BF cost/event")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6)
    ax.set_ylabel("Days per event (log scale)")
    ax.set_title("Computational Cost: FFS vs Brute Force", fontsize=10, fontweight="bold", pad=10)
    ax.legend(fontsize=7, framealpha=0.3)
    ax.grid(True, alpha=0.3)


def plot_outcome_breakdown(ax, df):
    """Stacked bar of success/failure/extratropical fractions per interface. FFS format only."""
    n_iface = get_n_interfaces(df)
    outcomes = [get_outcome_data(df, i) for i in range(n_iface)]

    if all(o is None for o in outcomes):
        _no_data(ax, "Outcome breakdown not available\n(needs success/failure/extratropical columns)")
        ax.set_title("λ-Interface Outcome Breakdown", fontsize=10, fontweight="bold", pad=10)
        return

    iface_labels = [f"λ{i}" for i in range(n_iface)]
    success_frac, fail_frac, extra_frac = [], [], []
    for o in outcomes:
        if o is None:
            success_frac.append(0); fail_frac.append(0); extra_frac.append(0)
        else:
            tot = o["success"] + o["failure"] + o["extratropical"]
            success_frac.append(o["success"] / tot * 100 if tot > 0 else 0)
            fail_frac.append(o["failure"] / tot * 100 if tot > 0 else 0)
            extra_frac.append(o["extratropical"] / tot * 100 if tot > 0 else 0)

    xpos = np.arange(n_iface)
    w = 0.5
    ax.bar(xpos, success_frac, w, color=COLORS["green"], alpha=0.8, label="Success")
    ax.bar(xpos, fail_frac, w, bottom=success_frac, color=COLORS["red"], alpha=0.6, label="Failure")
    ax.bar(xpos, extra_frac, w, bottom=np.array(success_frac) + np.array(fail_frac),
           color=COLORS["dim"], alpha=0.5, label="Extratropical")

    ax.set_xticks(xpos)
    ax.set_xticklabels(iface_labels)
    ax.set_ylabel("% of attempts")
    ax.set_ylim(0, 100)
    ax.set_title("λ-Interface Outcome Breakdown (Avg)", fontsize=10, fontweight="bold", pad=10)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.3)
    ax.grid(True, axis="y", alpha=0.3)


def plot_speedup_vs_directrate(ax, df):
    """Scatter of speedup vs direct formation rate. FFS format only."""
    if "direct_formation_rate_per_day" not in df.columns or "ffs_speedup" not in df.columns:
        _no_data(ax, "Needs direct_formation_rate_per_day & ffs_speedup")
        ax.set_title("Speedup vs Direct Rate", fontsize=10, fontweight="bold", pad=10)
        return

    ax.scatter(df["direct_formation_rate_per_day"], df["ffs_speedup"],
               c=COLORS["accent"], s=50, alpha=0.8, edgecolors="white", linewidth=0.5)

    z = np.polyfit(df["direct_formation_rate_per_day"], df["ffs_speedup"], 1)
    p = np.poly1d(z)
    xfit = np.linspace(df["direct_formation_rate_per_day"].min(), df["direct_formation_rate_per_day"].max(), 50)
    ax.plot(xfit, p(xfit), color=COLORS["orange"], ls="--", lw=1.2, alpha=0.7)

    corr = df["direct_formation_rate_per_day"].corr(df["ffs_speedup"])
    ax.set_xlabel("Direct formation rate (day⁻¹)")
    ax.set_ylabel("FFS Speedup")
    ax.set_title(f"Speedup vs Direct Rate (r = {corr:.2f})", fontsize=10, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3)


def plot_shooting_budget(ax, df, xlabels=None):
    """Stacked area of flux time vs shooting time. FFS format only."""
    if xlabels is None:
        xlabels = short_labels(df)
    x = np.arange(len(df))

    has_flux = "flux_total_time_days" in df.columns
    has_shoot = "shooting_time_days" in df.columns

    if not has_flux and not has_shoot:
        if "total_sim_days" in df.columns:
            ax.fill_between(x, 0, df["total_sim_days"], color=COLORS["accent"], alpha=0.4, label="Total sim time")
            ax.legend(fontsize=7, framealpha=0.3)
        else:
            _no_data(ax, "No simulation time columns found")
            ax.set_title("Computational Budget", fontsize=10, fontweight="bold", pad=10)
            return
    else:
        flux = df["flux_total_time_days"].values if has_flux else np.zeros(len(df))
        shoot = df["shooting_time_days"].values if has_shoot else np.zeros(len(df))
        ax.fill_between(x, 0, flux, color=COLORS["accent"], alpha=0.4, label="Flux sim time")
        ax.fill_between(x, flux, flux + shoot, color=COLORS["purple"], alpha=0.4, label="Shooting time")
        ax.plot(x, flux + shoot, color=COLORS["pink"], lw=1.2, alpha=0.7, label="Total FFS time")
        ax.legend(fontsize=7, framealpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6)
    ax.set_ylabel("Simulation days")
    ax.set_title("Computational Budget Breakdown", fontsize=10, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3)


def plot_lambda_crossings(ax, df, xlabels=None):
    """Bar/line chart of lambda crossings per IC window. Works for both formats."""
    if xlabels is None:
        xlabels = short_labels(df)
    x = np.arange(len(df))
    fmt = df.attrs.get("format", detect_format(df))
    n_iface = get_n_interfaces(df)

    if fmt == "ifs":
        for i in range(n_iface):
            col = f"n_crossed_lambda{i}"
            if col in df.columns and df[col].notna().any():
                ax.plot(x, df[col], color=LAMBDA_COLORS[i % len(LAMBDA_COLORS)],
                        marker="o", ms=3, lw=1.5, label=f"λ{i}")
    else:
        if "flux_lambda0_crossings" in df.columns:
            ax.plot(x, df["flux_lambda0_crossings"], color=LAMBDA_COLORS[0],
                    marker="o", ms=3, lw=1.5, label="λ0 crossings")
        if "n_stateB_arrivals" in df.columns:
            ax.plot(x, df["n_stateB_arrivals"], color=LAMBDA_COLORS[2],
                    marker="s", ms=3, lw=1.5, label="State B arrivals")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6)
    ax.set_ylabel("Count")
    ax.set_title("Lambda Crossings per IC Window", fontsize=10, fontweight="bold", pad=10)
    ax.legend(fontsize=7, framealpha=0.3)
    ax.grid(True, alpha=0.3)


def plot_mean_time_to_B(ax, df, xlabels=None):
    """Mean time to state B (IFS format)."""
    if xlabels is None:
        xlabels = short_labels(df)
    x = np.arange(len(df))

    if "mean_time_to_B_days" in df.columns:
        vals = df["mean_time_to_B_days"]
        ylabel = "Mean time to B (days)"
    elif "mean_time_to_B_hours" in df.columns:
        vals = df["mean_time_to_B_hours"]
        ylabel = "Mean time to B (hours)"
    else:
        _no_data(ax, "No mean_time_to_B column found")
        ax.set_title("Mean Time to State B", fontsize=10, fontweight="bold", pad=10)
        return

    valid = vals.notna()
    if valid.any():
        ax.plot(x[valid], vals[valid], color=COLORS["accent"], marker="o", ms=4, lw=1.8)
        ax.fill_between(x[valid], vals[valid], alpha=0.15, color=COLORS["accent"])
    else:
        _no_data(ax, "All mean_time_to_B values are NaN")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6)
    ax.set_ylabel(ylabel)
    ax.set_title("Mean Time to State B", fontsize=10, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3)


def plot_cps_rejected(ax, df, xlabels=None):
    """Plot CPS rejected counts (IFS format: n_cps_rejected)."""
    if xlabels is None:
        xlabels = short_labels(df)
    x = np.arange(len(df))

    if "n_cps_rejected" not in df.columns:
        _no_data(ax, "n_cps_rejected column not found")
        ax.set_title("CPS Rejected", fontsize=10, fontweight="bold", pad=10)
        return

    ax.bar(x, df["n_cps_rejected"], color=COLORS["orange"], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6)
    ax.set_ylabel("Count")
    ax.set_title("CPS Rejected per IC Window", fontsize=10, fontweight="bold", pad=10)
    ax.grid(True, axis="y", alpha=0.3)


# ── Dashboard builders ───────────────────────────────────────────────────────

def dashboard_ffs(df: pd.DataFrame, figsize=(18, 22), title=None):
    """Build the full 8-panel dashboard for FFS-format data."""
    apply_style()
    xlabels = short_labels(df)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title or "FFS Simulation Dashboard",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(4, 2, hspace=0.4, wspace=0.3,
                           left=0.06, right=0.94, top=0.95, bottom=0.03)

    plot_rate_evolution(fig.add_subplot(gs[0, 0]), df, xlabels)
    plot_speedup(fig.add_subplot(gs[0, 1]), df, xlabels)
    plot_pforward(fig.add_subplot(gs[1, 0]), df, xlabels)
    plot_trajectories(fig.add_subplot(gs[1, 1]), df, xlabels)
    plot_cost_comparison(fig.add_subplot(gs[2, 0]), df, xlabels)
    plot_outcome_breakdown(fig.add_subplot(gs[2, 1]), df)
    plot_speedup_vs_directrate(fig.add_subplot(gs[3, 0]), df)
    plot_shooting_budget(fig.add_subplot(gs[3, 1]), df, xlabels)

    return fig


def dashboard_ifs(df: pd.DataFrame, figsize=(18, 18), title=None):
    """Build a dashboard for IFS-format data."""
    apply_style()
    xlabels = short_labels(df)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title or "IFS Simulation Dashboard",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(4, 2, hspace=0.4, wspace=0.3,
                           left=0.06, right=0.94, top=0.93, bottom=0.04)

    plot_rate_evolution(fig.add_subplot(gs[0, 0]), df, xlabels)
    plot_pforward(fig.add_subplot(gs[0, 1]), df, xlabels)
    plot_trajectories(fig.add_subplot(gs[1, 0]), df, xlabels)
    plot_lambda_crossings(fig.add_subplot(gs[1, 1]), df, xlabels)
    plot_mean_time_to_B(fig.add_subplot(gs[2, 0]), df, xlabels)
    plot_cps_rejected(fig.add_subplot(gs[2, 1]), df, xlabels)
    plot_shooting_budget(fig.add_subplot(gs[3, 0]), df, xlabels)

    # Last slot: leave blank or add a custom panel
    ax_extra = fig.add_subplot(gs[3, 1])
    _no_data(ax_extra, "Available for custom plot")
    ax_extra.set_title("(Custom)", fontsize=10, fontweight="bold", pad=10)

    return fig


def dashboard_auto(df: pd.DataFrame, **kwargs):
    """Auto-detect format and build the appropriate dashboard."""
    fmt = df.attrs.get("format", detect_format(df))
    if fmt == "ffs":
        return dashboard_ffs(df, **kwargs)
    else:
        return dashboard_ifs(df, **kwargs)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize FFS / IFS simulation data")
    parser.add_argument("--ffs", type=str, default=None, help="Path to FFS-format TSV")
    parser.add_argument("--ifs", type=str, default=None, help="Path to IFS-format TSV")
    parser.add_argument("--file", type=str, default=None, help="Path to auto-detect format TSV")
    parser.add_argument("--output", type=str, default=None, help="Save figure to file")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    args = parser.parse_args()

    figs = []

    if args.file:
        df = load_data(args.file)
        figs.append(dashboard_auto(df))
    if args.ffs:
        df_ffs = load_data(args.ffs)
        figs.append(dashboard_ffs(df_ffs))
    if args.ifs:
        df_ifs = load_data(args.ifs)
        figs.append(dashboard_ifs(df_ifs))

    if not figs:
        print("No data files provided. Use --file, --ffs, or --ifs.")
        return

    if args.output:
        for i, fig in enumerate(figs):
            out = args.output if len(figs) == 1 else f"{Path(args.output).stem}_{i}{Path(args.output).suffix}"
            fig.savefig(out, dpi=args.dpi, facecolor=fig.get_facecolor())
            print(f"Saved to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()