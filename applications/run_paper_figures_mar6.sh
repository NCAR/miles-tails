#!/bin/bash
# ============================================================
# Regenerate all manuscript figures from results_mar6 data.
#
# Run from: /glade/derecho/scratch/schreck/FFS/
# Environment: conda activate credit-casper
#
# GPU required for steps 7 and 8 (model rollouts).
# All other steps are CPU-only.
#
# All figures land flat in PLOT_ROOT — LaTeX graphicspath is
# just {{./}{../plots/}}.
# ============================================================

set -e

APPS=/glade/work/schreck/repos/miles-tails/applications
FFS_SCRATCH=/glade/derecho/scratch/schreck/FFS
FFS_CONFIG=${FFS_SCRATCH}/ffs.yml
MODEL_CONFIG=${FFS_SCRATCH}/model.yml
RESULTS=${FFS_SCRATCH}/results_mar6
FFS_CSV=${RESULTS}/ffs_statistics_all_ics.csv
IFS_CSV=${RESULTS}/IFS/ifs_rates_FFS.csv
OUTPUT_DIR=${RESULTS}
PLOT_ROOT=/glade/work/schreck/repos/miles-tails/plots
CASE_IC="2022-09-02 00:00:00"
CASE_IC_DIR=${RESULTS}/2022-09-02T00Z

mkdir -p ${PLOT_ROOT}

# Single-IC CSV for Earl-specific figures
EARL_CSV=/tmp/ffs_earl_only.csv
python3 -c "
import pandas as pd
df = pd.read_csv('${FFS_CSV}')
df[df['time_label'] == '2022-09-02T00Z'].to_csv('${EARL_CSV}', index=False)
print(f'Earl CSV: {len(df[df[\"time_label\"] == \"2022-09-02T00Z\"])} row(s)')
"

# ============================================================
# 1. Rates plot + interface probs  (all ICs)
# ============================================================
echo "=== [1] plot_rates_comparison.py ==="
python ${APPS}/plot_rates_comparison.py \
    --ffs_config ${FFS_CONFIG} \
    --ffs_csv    ${FFS_CSV} \
    --ifs_csv    ${IFS_CSV} \
    --plot_dir   ${PLOT_ROOT}

# ============================================================
# 2. Commitment curve  (all ICs)
# ============================================================
echo "=== [2] plot_commitment_curve.py ==="
python ${APPS}/plot_commitment_curve.py \
    --ffs_config ${FFS_CONFIG} \
    --ffs_csv    ${FFS_CSV} \
    --ifs_csv    ${IFS_CSV} \
    --plot_dir   ${PLOT_ROOT}

# ============================================================
# 3. Reactive trajectories  (Earl IC only)
# ============================================================
echo "=== [3] plot_reactive_trajectories.py ==="
python ${APPS}/plot_reactive_trajectories.py \
    --ffs_config ${FFS_CONFIG} \
    --ffs_csv    ${EARL_CSV} \
    --output_dir ${OUTPUT_DIR} \
    --plot_dir   ${PLOT_ROOT} \
    --workers    8

# ============================================================
# 4. FFS tree  (Earl IC, top 5)
# ============================================================
echo "=== [4] plot_ffs_tree.py ==="
python ${APPS}/plot_ffs_tree.py \
    --ffs_config ${FFS_CONFIG} \
    --ic_dir     ${CASE_IC_DIR} \
    --plot_dir   ${PLOT_ROOT} \
    --top_k      5

# ============================================================
# 5. Committor map  (Earl IC only)
# ============================================================
echo "=== [5] plot_committor_map.py ==="
python ${APPS}/plot_committor_map.py \
    --ffs_config  ${FFS_CONFIG} \
    --ffs_csv     ${EARL_CSV} \
    --output_dir  ${OUTPUT_DIR} \
    --plot_dir    ${PLOT_ROOT} \
    --workers     8

# ============================================================
# 6. Committor fields  (all ICs aggregated)
# ============================================================
echo "=== [6] plot_committor_fields.py ==="
python ${APPS}/plot_committor_fields.py \
    --ffs_config ${FFS_CONFIG} \
    --ffs_csv    ${FFS_CSV} \
    --output_dir ${OUTPUT_DIR} \
    --plot_dir   ${PLOT_ROOT} \
    --workers    8 \
    --min_samples 5 \
    --n_bins     15

# ============================================================
# 7. Trajectory physics  (GPU — Earl IC)
# ============================================================
echo "=== [7a] plot_trajectory_physics.py --list_pathways ==="
python ${APPS}/plot_trajectory_physics.py \
    --model_config ${MODEL_CONFIG} \
    --ffs_config   ${FFS_CONFIG} \
    --ic_time      "${CASE_IC}" \
    --list_pathways

echo "=== [7b] plot_trajectory_physics.py (pathway 0 — update idx as needed) ==="
python ${APPS}/plot_trajectory_physics.py \
    --model_config ${MODEL_CONFIG} \
    --ffs_config   ${FFS_CONFIG} \
    --ic_time      "${CASE_IC}" \
    --pathway_idx  0 \
    --plot_dir     ${PLOT_ROOT}

# ============================================================
# 8. Physics composites  (GPU — Earl IC)
# ============================================================
echo "=== [8] plot_physics_composites.py ==="
python ${APPS}/plot_physics_composites.py \
    --model_config ${MODEL_CONFIG} \
    --ffs_config   ${FFS_CONFIG} \
    --output_dir   ${OUTPUT_DIR} \
    --plot_dir     ${PLOT_ROOT} \
    --workers      4

echo "=== DONE ==="
echo "All figures in: ${PLOT_ROOT}"
