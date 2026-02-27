# MILES-TAILS: Rare Event Sampling for AI Weather Prediction

<p align="center">
  <img src="images/tails.png" alt="MILES-TAILS" width="600"/>
</p>

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Forward Flux Sampling for hurricane genesis probability estimation in AI weather models. Multi-interface importance sampling with tropical cyclone tracking to efficiently compute rare event rates. Features parallel execution, extratropical filtering, MSLP-based detection, and comprehensive logging.

## Overview

This package implements Forward Flux Sampling algorithms to estimate the probability of rare atmospheric events—like hurricane genesis—in AI-based weather forecasting systems. Standard Monte Carlo sampling requires millions of trajectories to observe even a single hurricane formation. FFS provides a 10-1000× speedup through importance sampling along interfaces.

**Current implementation:** Atlantic basin hurricane genesis probability estimation from AI weather model forecasts.

## Rare Events in Complex Systems

<p align="center">
  <img src="images/rare_event_examples.png" alt="Rare Event Examples" width="800"/>
</p>

Forward Flux Sampling was developed to compute transition rates for rare events in molecular simulations. Examples of rare phenomena span molecular-scale processes like DNA duplex formation and ice nucleation, to large-scale atmospheric events like thunderstorms and hurricanes. In each case, these transitions occur too infrequently for direct simulation, making importance sampling essential for quantitative analysis. Here we apply FFS to hurricane genesis in AI weather models.

## Key Features

- **Forward Flux Sampling Framework**: Multi-interface importance sampling with automatic flux estimation
- **Multi-Storm Tracking**: Simultaneous detection and tracking of multiple tropical cyclones
- **Tropical Cyclone Classification**: Extratropical filtering with latitude/longitude rules
- **Parallel Execution**: Multi-GPU support with thread-safe configuration management
- **Real-time Visualization**: Optional live MSLP plotting during trajectory evolution
- **Production-Ready**: Comprehensive logging, verification checks, and visualization output

## Installation

```bash
git clone https://github.com/NCAR/miles-tails.git
cd miles-tails
pip install -e .
```

**Requirements:**
- Python 3.9+
- PyTorch
- xarray, numpy, scipy
- Cartopy (for visualization)
- CREDIT framework (for AI weather models)

## Quick Start

```python
import numpy as np
from tails.hurricane_genesis_ffs import HurricaneGenesisFFS
from credit.models import load_model
from credit.transforms import Normalize_ERA5_and_Forcing
from credit.datasets import Predict_Dataset_Batcher
from credit.datasets.load_dataset_and_dataloader import BatchForecastLenDataLoader

# Load your AI weather model
model = load_model(config, load_weights=True).to('cuda')

# Create initial dataset and loader
forecast_times = [['2022-09-01 00:00:00', '2022-09-11 00:00:00']]
dataset_params = {
    'zarr_path': '/path/to/data.zarr',
    'variables': config['data']['variables'],
    # ... other dataset parameters
}

initial_dataset = Predict_Dataset_Batcher(
    **dataset_params,
    fcst_datetime=forecast_times,
)
initial_loader = BatchForecastLenDataLoader(initial_dataset)

# Initialize FFS with tracking
ffs = HurricaneGenesisFFS(
    model=model,
    state_transformer=Normalize_ERA5_and_Forcing(config),
    config=config,
    initial_dataset=initial_dataset,
    dataset_params=dataset_params,
    interfaces=[1000, 988, 980, 975, 970],  # Progressive intensification
    state_A=1008,           # No organized system (hPa)
    state_B=965,            # Hurricane strength (hPa)
    output_dir='./results',
    rank=0,
    world_size=1,
    worker_id=0,
    use_cps=False           # Set True for extratropical filtering
)

# Enable visualization (optional - shows live MSLP plots)
ffs.enable_visualization()

# Run FFS algorithm
ffs.run_ffs(
    initial_loader=initial_loader,
    n_flux_trials=100,      # Flux generation trajectories
    n_shoot_trials=50       # Shooting attempts per interface
)

# Disable visualization when done
ffs.disable_visualization()

# Results
print(f"Hurricane genesis rate: {ffs.flux_estimate * np.prod(ffs.transition_probs):.2e} per day")
```

---

## Running FFS on HPC Systems

All production runs use `applications/run_parallel_ffs.py` with two config files.

### Configuration Files

**`ffs.yml`** — FFS algorithm parameters:
```yaml
forecast_times:
  - ['2022-08-21 00:00:00', '2022-09-10 00:00:00']
  - ['2022-08-22 00:00:00', '2022-09-11 00:00:00']
  # ... more initial conditions

single_ic_mode: true      # All GPUs work on one IC at a time
ic_index: 0               # Which IC to process (overridden by job array)

state_A: 1008
state_B: 965
interfaces: [1000, 988, 980, 975, 970]
use_cps: false            # Set to true for extratropical filtering

n_flux_total: 1000        # Total flux trajectories across all workers
n_shoot_per_interface: 500  # Shooting attempts per interface
n_workers: 2              # Workers per GPU

output_dir: './results'
```

**`model.yml`** — AI weather model configuration (CREDIT framework)

---

### Step 1 — Flux Generation

Runs long trajectories from state A to estimate the crossing rate Φ₀ at the first interface λ₀.

```bash
# Single IC, single GPU
python applications/run_parallel_ffs.py \
    --model_config model.yml \
    --ffs_config ffs.yml \
    --phase flux \
    --ic_index 0

# Multi-GPU (4 GPUs, 2 workers each)
for gpu in {0..3}; do
    CUDA_VISIBLE_DEVICES=${gpu} \
    torchrun --nproc_per_node=1 --master-port=$((RANDOM % 10000 + 20000)) \
        applications/run_parallel_ffs.py \
        --model_config model.yml \
        --ffs_config ffs.yml \
        --phase flux \
        --num_workers 2 \
        --ic_index 0 &
done
wait
```

Output appears in `results/2022-08-21T00Z/flux/` (configs + PNGs) and `results/2022-08-21T00Z/logs/flux/` (JSONL logs).

---

### Step 2 — Shooting Phases

Run sequentially, one interface at a time. Each phase reads configs from the previous interface and shoots toward the next.

```bash
# λ₀ → λ₁  (interface=0)
python applications/run_parallel_ffs.py \
    --model_config model.yml \
    --ffs_config ffs.yml \
    --phase shoot \
    --interface 0 \
    --ic_index 0

# λ₁ → λ₂  (interface=1)
python applications/run_parallel_ffs.py \
    --model_config model.yml \
    --ffs_config ffs.yml \
    --phase shoot \
    --interface 1 \
    --ic_index 0

# λ₂ → λ₃, λ₃ → λ₄  (interfaces=2,3)
# ... repeat with --interface 2, --interface 3
```

| `--interface` | Reads from | Shoots to |
|:---:|---|---|
| 0 | `flux/` (λ₀ configs) | `1/` (λ₁ configs) |
| 1 | `1/` (λ₁ configs) | `2/` (λ₂ configs) |
| 2 | `2/` | `3/` |
| 3 | `3/` | `4/` |

---

### Job Submission Script (PBS/Derecho)

```bash
#!/bin/bash
#PBS -A PROJECT_CODE
#PBS -N hurricane_ffs
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -q main

SCRIPT_DIR=/glade/work/schreck/repos/miles-tails/applications
MODEL_CONFIG=model.yml
FFS_CONFIG=ffs.yml

IC_INDEX=${PBS_ARRAY_INDEX:-0}
PHASE=flux           # Options: flux, shoot
INTERFACE=0          # Only used for shoot phase
NUM_WORKERS=2        # Workers per GPU

for gpu in {0..3}; do
    CUDA_VISIBLE_DEVICES=${gpu} \
    torchrun --nproc_per_node=1 --master-port=$((RANDOM % 10000 + 20000)) \
        ${SCRIPT_DIR}/run_parallel_ffs.py \
        --model_config ${MODEL_CONFIG} \
        --ffs_config ${FFS_CONFIG} \
        --phase ${PHASE} \
        --interface ${INTERFACE} \
        --num_workers ${NUM_WORKERS} \
        --ic_index ${IC_INDEX} &
done
wait
```

**Job arrays** — process all ICs automatically:
```bash
#PBS -J 0-49   # Process ICs 0–49, one per job
# PBS_ARRAY_INDEX is automatically passed as ic_index
```

---

## Analysis Pipeline

Run these scripts in order after FFS is complete. All scripts are in `applications/` and should be run from the repo root.

### 1. Analyze FFS Logs

Parses JSONL logs across all IC directories, computes per-interface transition probabilities, and saves a summary CSV used by all downstream scripts.

```bash
# Trace all pathways to state B and generate ffs_statistics_all_ics.csv
python applications/analyze_ffs_logs.py ffs.yml --trace_all

# Trace pathways to a specific interface only (e.g. λ₁)
python applications/analyze_ffs_logs.py ffs.yml --target_interface 1
```

Output: `results/ffs_statistics_all_ics.csv` — one row per IC with flux, transition probabilities, and rates.

---

### 2. IFS Brute-Force Rates

Computes hurricane genesis rates directly from IFS ensemble forecasts using the same multi-storm tracking methodology as FFS. Used as a reference benchmark.

```bash
python applications/ifs_brute_force_rates.py \
    --ffs_config ffs.yml \
    --ifs_path /glade/derecho/scratch/schreck/IFS.zarr \
    --n_jobs 8
```

Output: `results/IFS/ifs_rates_FFS.csv`

---

### 3. Identify Reactive Pathways

Traces the complete genealogy from every state-B config back to its λ₀ seed, clusters correlated trajectories by their earliest branch point, and selects one independent representative per cluster. Saves a JSON and PNG for each reactive trajectory.

```bash
python applications/reactive_pathways.py ffs.yml

# Options
python applications/reactive_pathways.py ffs.yml \
    --min-branch-degree 2 \         # Min descendants to be a branch point (default: 2)
    --selection shortest \          # Representative selection: shortest | deepest_mslp | first
    --workers 8 \                   # Parallel workers for figure generation (default: min(8, ncpu))
    --no_plot                       # Skip figure output (JSON only — much faster)
```

Output per IC: `results/2022-08-21T00Z/reactive_trajectories/reactive_trajectories.json`

---

### 4. Plot Reactive Trajectories

Spaghetti track map of all reactive trajectories (left panel) and a 2D cluster-weighted crossing-density heatmap with transition flow arrows (right panel). One figure per IC.

```bash
python applications/plot_reactive_trajectories.py \
    --ffs_config ffs.yml \
    --ffs_csv    results/ffs_statistics_all_ics.csv \
    --output_dir results \
    --plot_dir   results/plots \
    --workers    8 \                # Parallel workers for load+plot (default: min(8, ncpu))
    --no_plot                       # Load tracks only, skip figure output
```

Output: `results/plots/reactive_trajectories/reactive_trajectories_YYYY-MM-DD.png`

---

### 5. Plot FFS Tree

Draws the complete forward branching tree from a single λ₀ seed — all shooting attempts at every interface — on a zoomed Atlantic map. Useful as an explainer figure for papers and presentations.

```bash
# Auto-select the IC and λ₀ with the most state-B descendants (parallel scan)
python applications/plot_ffs_tree.py \
    --ffs_config ffs.yml \
    --ffs_csv    results/ffs_statistics_all_ics.csv \
    --output_dir results \
    --plot_dir   results/plots \
    --workers    8              # Parallel workers for multi-IC log scanning (default: min(8, ncpu))

# Single IC — auto-select best λ₀
python applications/plot_ffs_tree.py \
    --ffs_config ffs.yml \
    --ic_dir     results/2022-08-21T00Z \
    --plot_dir   results/plots

# Rank all λ₀ roots by B-descendant count (inspect before plotting)
python applications/plot_ffs_tree.py \
    --ffs_config ffs.yml \
    --ic_dir     results/2022-08-21T00Z \
    --rank

# Manually specify a λ₀ root
python applications/plot_ffs_tree.py \
    --ffs_config ffs.yml \
    --ic_dir     results/2022-08-21T00Z \
    --root       lambda0_config_1049_EN \
    --plot_dir   results/plots
```

Output: `results/plots/ffs_tree_YYYY-MM-DD_lambda0_config_XXXX_YY.png`

---

### 6. Plot Commitment Curve

Plots the committor function p_B(λᵢ) — the probability of reaching state B given that interface λᵢ has been crossed — for both FFS (product of forward transition probabilities) and IFS (brute-force count ratios).

```bash
python applications/plot_commitment_curve.py \
    --ffs_config ffs.yml \
    --ffs_csv    results/ffs_statistics_all_ics.csv \
    --ifs_csv    results/IFS/ifs_rates_FFS.csv \
    --plot_dir   results/plots
```

Output: `results/plots/commitment_curve.png`

---

## Full Pipeline Summary

```bash
# 1. Run FFS (flux + 4 shooting phases per IC)
python applications/run_parallel_ffs.py --model_config model.yml --ffs_config ffs.yml --phase flux
python applications/run_parallel_ffs.py --model_config model.yml --ffs_config ffs.yml --phase shoot --interface 0
python applications/run_parallel_ffs.py --model_config model.yml --ffs_config ffs.yml --phase shoot --interface 1
python applications/run_parallel_ffs.py --model_config model.yml --ffs_config ffs.yml --phase shoot --interface 2
python applications/run_parallel_ffs.py --model_config model.yml --ffs_config ffs.yml --phase shoot --interface 3

# 2. Analyze logs → statistics CSV
python applications/analyze_ffs_logs.py ffs.yml --trace_all

# 3. IFS reference rates
python applications/ifs_brute_force_rates.py --ffs_config ffs.yml --ifs_path /path/to/IFS.zarr --n_jobs 8

# 4. Reactive pathways (--no_plot to skip figures and only write JSON)
python applications/reactive_pathways.py ffs.yml --workers 8

# 5. Plots
python applications/plot_reactive_trajectories.py --ffs_config ffs.yml --ffs_csv results/ffs_statistics_all_ics.csv --output_dir results --plot_dir results/plots --workers 8
python applications/plot_ffs_tree.py              --ffs_config ffs.yml --ffs_csv results/ffs_statistics_all_ics.csv --output_dir results --plot_dir results/plots --workers 8
python applications/plot_commitment_curve.py       --ffs_config ffs.yml --ffs_csv results/ffs_statistics_all_ics.csv --ifs_csv results/IFS/ifs_rates_FFS.csv --plot_dir results/plots
```

---

## Algorithm Overview

Forward Flux Sampling estimates rare event rates by:

1. **Flux Generation (Phase 0)**: Long trajectories from state A to estimate crossing rate Φ₀ at first interface λ₀
2. **Shooting (Phases 1-N)**: Short trajectories from each interface λᵢ to estimate transition probabilities P(λᵢ→λᵢ₊₁)
3. **Rate Calculation**: P(A→B) = Φ₀ × ∏ P(λᵢ→λᵢ₊₁)

**Order Parameter:** Mean sea level pressure (MSLP)
- State A: MSLP > 1008 hPa (quiescent)
- Interfaces: 1000, 988, 980, 975, 970 hPa
- State B: MSLP < 965 hPa (hurricane)

### Multi-Storm Tracking

The flux generation phase tracks multiple storms simultaneously:
- Tempest-style local minimum detection (3×3 neighborhood test)
- Automatic merging of nearby minima (< 10° separation)
- Independent tracking until dissipation (MSLP > A)
- Only NEW genesis events counted (decorrelation via state A return)

### Extratropical Filtering

Storms are rejected if:
- Latitude > 50°N (too far north)
- Longitude > -10°W (heading to Europe)
- Latitude > 30°N AND over land (landfall)

This ensures only tropical cyclones are counted.

---

## Directory Structure

```
miles-tails/
├── tails/
│   ├── hurricane_genesis_ffs.py      # Core FFS engine
│   ├── ffs_logger.py                 # JSONL logging
│   └── cyclone_phase_tracker.py      # CPS classification (optional)
├── applications/
│   ├── run_parallel_ffs.py           # Multi-GPU parallel FFS execution
│   ├── analyze_ffs_logs.py           # Parse logs → statistics CSV
│   ├── reactive_pathways.py          # Identify independent reactive trajectories
│   ├── ifs_brute_force_rates.py      # IFS ensemble reference rates
│   ├── plot_reactive_trajectories.py # Spaghetti tracks + density heatmap
│   ├── plot_ffs_tree.py              # Single-seed branching tree figure
│   └── plot_commitment_curve.py      # Committor p_B(λᵢ) vs interface
├── config/
│   ├── ffs.yml                       # FFS algorithm configuration
│   └── sdl_wxformer.yml              # Model configuration
└── README.md
```

## Output Structure

```
results/
├── ffs_statistics_all_ics.csv        # Combined statistics across all ICs
├── IFS/
│   └── ifs_rates_FFS.csv             # IFS brute-force reference rates
├── plots/
│   ├── reactive_trajectories/
│   │   └── reactive_trajectories_YYYY-MM-DD.png
│   ├── ffs_tree_YYYY-MM-DD_lambda0_config_XXXX_YY.png
│   └── commitment_curve.png
└── 2022-08-21T00Z/                   # One directory per initial condition
    ├── logs/
    │   ├── flux/                     # Flux generation JSONL logs
    │   ├── 1/                        # Shooting logs λ₀→λ₁
    │   ├── 2/                        # Shooting logs λ₁→λ₂
    │   └── .../
    ├── flux/                         # λ₀ crossing configs + PNGs
    ├── 1/                            # λ₁ configs
    ├── 2/                            # λ₂ configs
    ├── stateB/                       # Hurricane formation configs
    └── reactive_trajectories/
        ├── reactive_trajectories.json
        └── reactive_trajectory_NNN.png
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{miles_tails_ffs,
  author = {John Schreck and MILES Group},
  title = {MILES-TAILS: Rare Event Sampling for AI Weather Prediction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/NCAR/miles-tails}
}
```

## References

- Allen, R. J., Warren, P. B., & ten Wolde, P. R. (2005). *Sampling rare switching events in biochemical networks*. Physical Review Letters, 94(1), 018104.
- FFS methodology adapted for atmospheric science applications

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Developed by the [MILES](https://www.cisl.ucar.edu/miles) (Machine Intelligence Learning for Earth System) group at NCAR.

This work builds on the [CREDIT](https://github.com/NCAR/miles-credit) framework for AI weather prediction.

## Contact

- **Author**: John Schreck
- **Institution**: National Center for Atmospheric Research (NCAR)
- **Group**: MILES
