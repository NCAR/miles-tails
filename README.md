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
    output_dir='./ffs_results',
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

## Parallel Execution on HPC Systems

For production runs processing multiple initial conditions across many interfaces, use the parallel execution script with job schedulers.

### Configuration Files

**ffs.yml** - FFS algorithm parameters:
```yaml
forecast_times:
  - ['2022-08-21 00:00:00', '2022-09-10 00:00:00']
  - ['2022-08-22 00:00:00', '2022-09-11 00:00:00']
  # ... more initial conditions

single_ic_mode: true      # All GPUs work on one IC at a time
ic_index: 0               # Which IC to process (can be overridden by job array)

state_A: 1008
state_B: 965
interfaces: [1000, 988, 980, 975, 970]
use_cps: false            # Set to true for extratropical filtering

n_flux_total: 1000        # Total flux trajectories across all workers
n_shoot_per_interface: 500  # Shooting attempts per interface
n_workers: 2              # Workers per GPU

output_dir: './results'
```

**model.yml** - AI weather model configuration (CREDIT framework)

### Job Submission Script

```bash
#!/bin/bash
#PBS -A PROJECT_CODE
#PBS -N hurricane_ffs
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -q main

# ===== JOB CONFIGURATION =====
SCRIPT_DIR=/path/to/miles-tails/applications
FFS_SCRIPT=${SCRIPT_DIR}/run_parallel_ffs.py
MODEL_CONFIG=model.yml
FFS_CONFIG=ffs.yml

IC_INDEX=${PBS_ARRAY_INDEX:-0}  # Use job array index if available

# ===== CONFIGURE PHASE HERE =====
PHASE=flux           # Options: flux, shoot
INTERFACE=0          # Only used for shoot phase
                     # 0 = shoot from λ₀ to λ₁
                     # 1 = shoot from λ₁ to λ₂
                     # etc.
NUM_WORKERS=2        # Workers per GPU

# Launch 4 parallel processes (1 per GPU)
if [ "${PHASE}" = "flux" ]; then
    for gpu in {0..3}; do
        CUDA_VISIBLE_DEVICES=${gpu} \
        torchrun --nproc_per_node=1 --master-port=$((RANDOM % 10000 + 20000)) \
            ${FFS_SCRIPT} \
            --model_config ${MODEL_CONFIG} \
            --ffs_config ${FFS_CONFIG} \
            --phase flux \
            --num_workers ${NUM_WORKERS} \
            --ic_index ${IC_INDEX} &
    done
else
    for gpu in {0..3}; do
        CUDA_VISIBLE_DEVICES=${gpu} \
        torchrun --nproc_per_node=1 --master-port=$((RANDOM % 10000 + 20000)) \
            ${FFS_SCRIPT} \
            --model_config ${MODEL_CONFIG} \
            --ffs_config ${FFS_CONFIG} \
            --phase shoot \
            --interface ${INTERFACE} \
            --num_workers ${NUM_WORKERS} \
            --ic_index ${IC_INDEX} &
    done
fi
wait
```

### Execution Workflow

**1. Flux Generation**
```bash
# Edit job script: PHASE=flux
qsub job_script.sh
```

Generates λ₀ crossings. Monitor progress in `results/IC_YYYY-MM-DDTHH/flux/`

**2. Sequential Shooting Phases**
```bash
# Shoot from λ₀ → λ₁
# Edit job script: PHASE=shoot, INTERFACE=0
qsub job_script.sh

# After completion, shoot from λ₁ → λ₂
# Edit job script: PHASE=shoot, INTERFACE=1
qsub job_script.sh

# Continue for remaining interfaces...
# INTERFACE=2 for λ₂ → λ₃
# INTERFACE=3 for λ₃ → λ₄
```

**3. Analysis**
```bash
python analyze_ffs_logs.py ffs.yml --trace_all
```

### Processing Multiple Initial Conditions

**Option 1: Job Arrays** (one IC per job)
```bash
#PBS -J 0-49  # Process ICs 0-49

# Script automatically uses PBS_ARRAY_INDEX as ic_index
```

**Option 2: Single IC Mode** (all GPUs on one IC)
```yaml
# ffs.yml
single_ic_mode: true
ic_index: 5  # Process IC #5
```

Submit separate jobs for each IC by changing `ic_index` in ffs.yml.

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

## Directory Structure

```
miles-tails/
├── tails/
│   ├── __init__.py
│   ├── hurricane_genesis_ffs.py  # Main FFS implementation
│   ├── ffs_logger.py              # FFS logging utilities
│   └── cyclone_phase_tracker.py  # CPS classification (optional)
├── applications/
│   ├── run_parallel_ffs.py        # Multi-IC parallel execution
│   └── analyze_ffs_logs.py        # Results analysis
├── tests/
├── docs/
└── README.md
```

## Output Structure

```
results/
└── 2022-08-28T00Z/               # One directory per initial condition
    ├── logs/
    │   ├── flux/                  # Flux generation logs
    │   ├── 1/                     # Shooting logs (λ₀→λ₁)
    │   └── 2/                     # Shooting logs (λ₁→λ₂)
    ├── flux/                      # λ₀ crossings (configs + PNGs)
    ├── 1/                         # λ₁ crossings
    ├── 2/                         # λ₂ crossings
    └── stateB/                    # Hurricane formations
```

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