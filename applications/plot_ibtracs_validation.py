"""
plot_ibtracs_validation.py

Tier-1 external validation: compare FFS genesis rates against observed
2022 Atlantic tropical cyclogenesis activity from IBTrACS v04.

For each of the 98 FFS initial conditions, we count how many observed
Atlantic named storms had their genesis within the 15-day FFS simulation
window and within the FFS domain [10-40N, 100-20W].  We then compute
the Spearman rank correlation between k^FFS and this observed genesis count.

Outputs
-------
  <plot_dir>/ibtracs_validation.png  — 3-panel figure:
      (1) FFS rates with IBTrACS genesis events marked
      (2) Scatter: k^FFS vs. observed genesis count-in-window
      (3) Correlation summary text panel
  Prints Spearman rho and p-value to stdout.

Usage
-----
  python plot_ibtracs_validation.py \
      --all_ics_csv /glade/derecho/scratch/schreck/FFS/results_mar18/ffs_statistics_all_ics.csv \
      --plot_dir   /glade/derecho/scratch/schreck/FFS/results_mar18/plots/summary \
      [--ibtracs_csv /path/to/ibtracs.csv]   # optional; downloads if absent
"""

import argparse
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import spearmanr

# ── constants ─────────────────────────────────────────────────────────────────
IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-"
    "stewardship-ibtracs/v04r00/access/csv/ibtracs.NA.list.v04r00.csv"
)
IBTRACS_CACHE = Path("/glade/derecho/scratch/schreck/FFS") / "ibtracs_NA.csv"

FFS_WINDOW_DAYS = 15          # length of each FFS simulation window
DOMAIN_LAT = (10.0, 40.0)     # FFS domain (N)
DOMAIN_LON = (-100.0, -20.0)  # FFS domain (W, negative)

# Named storms to label on the time-series panel
LABEL_STORMS = {"EARL", "FIONA", "IAN", "JULIA", "DANIELLE"}


# ── IBTrACS helpers ───────────────────────────────────────────────────────────

def fetch_ibtracs(ibtracs_csv: Path | None) -> pd.DataFrame:
    """Return IBTrACS dataframe, downloading if necessary."""
    if ibtracs_csv is not None and Path(ibtracs_csv).exists():
        path = Path(ibtracs_csv)
    elif IBTRACS_CACHE.exists():
        path = IBTRACS_CACHE
    else:
        print(f"Downloading IBTrACS last-3-years CSV → {IBTRACS_CACHE} ...")
        IBTRACS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(IBTRACS_URL, IBTRACS_CACHE)
        path = IBTRACS_CACHE

    # IBTrACS CSV has 2 header rows; second row is units
    df = pd.read_csv(path, skiprows=[1], low_memory=False, na_values=[" ", ""])
    return df


def extract_genesis_events(df: pd.DataFrame, year: int = 2022) -> pd.DataFrame:
    """
    Return one row per named storm's first recorded position in the requested year.
    The NA-specific IBTrACS file is already filtered to North Atlantic basin.
    Excludes NOT_NAMED/unnamed entries.
    Columns: storm_name, genesis_time (UTC), lat, lon
    """
    df = df.copy()
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df = df[df["ISO_TIME"].dt.year == year].dropna(subset=["ISO_TIME"])

    records = []
    for sid, grp in df.groupby("SID"):
        grp = grp.sort_values("ISO_TIME")
        name = str(grp.iloc[0]["NAME"]).strip().upper()
        if not name or name.startswith("NOT") or name == "NAN":
            continue
        first = grp.iloc[0]
        try:
            lat = float(first["LAT"])
            lon = float(first["LON"])
        except (ValueError, TypeError):
            continue
        records.append({
            "storm_name": name,
            "genesis_time": first["ISO_TIME"],
            "lat": lat,
            "lon": lon,
        })

    genesis = pd.DataFrame(records)
    # Loose domain filter — western Atlantic, Caribbean, Gulf
    genesis = genesis[
        (genesis["lat"] >= 5) & (genesis["lat"] <= 50) &
        (genesis["lon"] >= -110) & (genesis["lon"] <= -15)
    ].copy()
    return genesis.sort_values("genesis_time").reset_index(drop=True)


def count_genesis_in_window(genesis: pd.DataFrame,
                             ic_time: pd.Timestamp,
                             window_days: int = FFS_WINDOW_DAYS,
                             lat_min: float = DOMAIN_LAT[0],
                             lat_max: float = DOMAIN_LAT[1],
                             lon_min: float = DOMAIN_LON[0],
                             lon_max: float = DOMAIN_LON[1]) -> int:
    """Count observed genesis events within [ic_time, ic_time + window_days] and domain."""
    t_end = ic_time + pd.Timedelta(days=window_days)
    mask = (
        (genesis["genesis_time"] >= ic_time) &
        (genesis["genesis_time"] < t_end) &
        (genesis["lat"] >= lat_min) & (genesis["lat"] <= lat_max) &
        (genesis["lon"] >= lon_min) & (genesis["lon"] <= lon_max)
    )
    return int(mask.sum())


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    # ── load FFS rates ─────────────────────────────────────────────────────
    rates = pd.read_csv(args.all_ics_csv)
    rates["ic_time"] = pd.to_datetime(rates["ic_time"])
    rates = rates.sort_values("ic_time").reset_index(drop=True)
    print(f"Loaded {len(rates)} ICs from {args.all_ics_csv}")

    # ── load IBTrACS ───────────────────────────────────────────────────────
    ibtracs_raw = fetch_ibtracs(args.ibtracs_csv)
    genesis = extract_genesis_events(ibtracs_raw, year=2022)
    print(f"\n2022 Atlantic genesis events in domain:")
    print(genesis[["storm_name", "genesis_time", "lat", "lon"]].to_string(index=False))

    # ── restrict genesis to the FFS season window ─────────────────────────
    # Only storms that formed within Aug 21 – Oct 23 (last IC + 15 days)
    season_start = rates["ic_time"].min()
    season_end   = rates["ic_time"].max() + pd.Timedelta(days=FFS_WINDOW_DAYS)
    genesis_season = genesis[
        (genesis["genesis_time"] >= season_start) &
        (genesis["genesis_time"] <= season_end)
    ].reset_index(drop=True)
    print(f"\nGenesis events within FFS season+window:")
    print(genesis_season[["storm_name", "genesis_time", "lat", "lon"]].to_string(index=False))

    # ── per-IC: lead time to NEXT genesis in domain ────────────────────────
    def lead_time_to_next(ic_t):
        """Days from ic_t to earliest genesis that falls within [ic_t, ic_t+15d]."""
        mask = (
            (genesis_season["genesis_time"] >= ic_t) &
            (genesis_season["genesis_time"] < ic_t + pd.Timedelta(days=FFS_WINDOW_DAYS))
        )
        hits = genesis_season[mask]
        if hits.empty:
            return np.nan
        return (hits["genesis_time"].min() - ic_t).total_seconds() / 86400.0

    rates["lead_days"] = rates["ic_time"].apply(lead_time_to_next)
    rates["lead_bin"] = pd.cut(
        rates["lead_days"],
        bins=[-0.01, 7, 15, np.inf],
        labels=["≤7 days\n(imminent)", "7–15 days\n(near-term)", ">15 days\n(no genesis)"],
    )

    # ── per-storm lead-time series ─────────────────────────────────────────
    # For each named storm: ICs within 15 days before genesis, coloured by storm
    focus_storms = ["DANIELLE", "EARL", "FIONA", "IAN", "JULIA"]
    storm_colors = {
        "DANIELLE": "tab:blue",
        "EARL":     "tab:orange",
        "FIONA":    "tab:green",
        "IAN":      "tab:red",
        "JULIA":    "tab:purple",
    }
    per_storm = {}
    for _, ev in genesis_season.iterrows():
        name = ev["storm_name"]
        if name not in focus_storms:
            continue
        t_gen = ev["genesis_time"]
        mask = (
            (rates["ic_time"] <= t_gen) &
            (rates["ic_time"] >= t_gen - pd.Timedelta(days=FFS_WINDOW_DAYS))
        )
        sub = rates[mask].copy()
        sub["lead_to_storm"] = (t_gen - sub["ic_time"]).dt.total_seconds() / 86400.0
        per_storm[name] = sub

    # ── Spearman correlations ──────────────────────────────────────────────
    log_rate = np.log10(rates["ffs_rate_per_day"])
    # (1) vs. raw count in window (original analysis)
    rates["obs_genesis_count"] = rates["ic_time"].apply(
        lambda t: count_genesis_in_window(genesis_season, t)
    )
    rho_count, p_count = spearmanr(log_rate, rates["obs_genesis_count"])
    # (2) vs. lead time (negative: shorter lead = higher rate expected)
    valid = rates["lead_days"].notna()
    rho_lead, p_lead = spearmanr(log_rate[valid], rates["lead_days"][valid])

    print(f"\nSpearman ρ(log10 k^FFS, obs_genesis_count) = {rho_count:.3f}  p = {p_count:.4f}")
    print(f"Spearman ρ(log10 k^FFS, lead_days_to_genesis) = {rho_lead:.3f}  p = {p_lead:.4f}  "
          f"[n={valid.sum()}, expect negative: shorter lead → higher k]")

    print(f"\nLead-time bin breakdown:")
    for label, grp in rates.groupby("lead_bin", observed=True):
        print(f"  {label!s:30s}  n={len(grp):3d}  "
              f"median log10(k)={np.log10(grp['ffs_rate_per_day']).median():.3f}  "
              f"range=[{np.log10(grp['ffs_rate_per_day']).min():.2f}, "
              f"{np.log10(grp['ffs_rate_per_day']).max():.2f}]")

    # ── figure: 2 panels ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 11),
                             gridspec_kw={"height_ratios": [3, 2.5]})

    # ── Panel 1: Rate time series + genesis markers ────────────────────────
    ax1 = axes[0]
    ax1.semilogy(rates["ic_time"], rates["ffs_rate_per_day"],
                 color="tab:green", lw=1.8, label=r"$k^{\rm FFS}$", zorder=3)
    ax1.semilogy(rates["ic_time"], rates["direct_formation_rate_per_day"],
                 color="tab:purple", lw=1.0, alpha=0.5, label=r"$k^{\rm direct}$", zorder=2)

    ymax = rates["ffs_rate_per_day"].max()
    ymin = rates["ffs_rate_per_day"].min()
    ax1.set_ylim(bottom=ymin * 0.3, top=ymax * 6)
    for _, ev in genesis_season.iterrows():
        name = ev["storm_name"]
        color = storm_colors.get(name, "gray")
        ax1.axvline(ev["genesis_time"], color=color, lw=1.5, ls="--", alpha=0.8, zorder=1)
        ax1.text(ev["genesis_time"], ymax * 1.8, name.capitalize(),
                 fontsize=11, ha="center", va="bottom", color=color,
                 fontweight="bold" if name in focus_storms else "normal",
                 rotation=40, clip_on=False)

    ax1.set_ylabel(r"Genesis rate $k$ (day$^{-1}$)", fontsize=11)
    ax1.legend(fontsize=9, loc="lower left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax1.grid(True, which="both", alpha=0.3)

    # ── Panel 2: k^FFS vs. observed genesis count scatter ─────────────────
    ax2 = axes[1]
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.08, 0.08, len(rates))
    sc = ax2.scatter(
        rates["obs_genesis_count"] + jitter,
        rates["ffs_rate_per_day"],
        c=mdates.date2num(rates["ic_time"].tolist()),
        cmap="plasma", s=45, alpha=0.85, edgecolors="k", linewidths=0.3, zorder=3,
    )
    cbar = plt.colorbar(sc, ax=ax2, pad=0.02)
    cbar.set_label("IC date", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda x, _: mdates.num2date(x).strftime("%b %d")
        )
    )
    ax2.set_yscale("log")
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["1", "2", "3"], fontsize=10)
    ax2.set_xlabel(
        "Observed named-storm genesis events in 15-day window\n"
        "(IBTrACS v04, North Atlantic, domain 10–40°N 100–20°W)",
        fontsize=10,
    )
    ax2.set_ylabel(r"$k^{\rm FFS}$ (day$^{-1}$)", fontsize=11)
    # Annotate the October cluster (high count, low rate = suppressed environment)
    oct_mask = rates["ic_time"] >= pd.Timestamp("2022-10-01")
    oct_sub = rates[oct_mask]
    if len(oct_sub):
        ax2.annotate(
            "Oct ICs: Julia/Karl form\ndespite suppressed k^FFS",
            xy=(oct_sub["obs_genesis_count"].mean(),
                oct_sub["ffs_rate_per_day"].median()),
            xytext=(oct_sub["obs_genesis_count"].mean() + 0.55, 2e-3),
            fontsize=8, color="navy",
            ha="left",
            arrowprops=dict(arrowstyle="->", color="navy", lw=1.2,
                            shrinkA=0, shrinkB=4),
        )
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout(pad=2.5)
    out_path = Path(args.plot_dir) / "ibtracs_validation.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")

    # ── clean summary table ────────────────────────────────────────────────
    print("\n── Per-IC lead time summary (sample) ────────────────────────")
    disp = rates[["ic_time", "ffs_rate_per_day", "lead_days", "lead_bin"]].copy()
    disp["log10_k"] = np.log10(disp["ffs_rate_per_day"]).round(3)
    print(disp[["ic_time", "log10_k", "lead_days", "lead_bin"]]
          .to_string(index=False, max_rows=30))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all_ics_csv",
        default="/glade/derecho/scratch/schreck/FFS/results_mar18/ffs_statistics_all_ics.csv",
    )
    parser.add_argument(
        "--plot_dir",
        default="/glade/derecho/scratch/schreck/FFS/results_mar18/plots/summary",
    )
    parser.add_argument(
        "--ibtracs_csv",
        default=None,
        help="Path to local IBTrACS CSV (downloads if not provided)",
    )
    args = parser.parse_args()
    main(args)
