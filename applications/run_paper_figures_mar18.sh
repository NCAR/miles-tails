#!/bin/bash
# ============================================================
# Regenerate all manuscript figures from results_mar18 data.
#
# Run from: /glade/derecho/scratch/schreck/FFS/
# Environment: source activate credit-main-casper
#
# Plots are organized into subdirectories under PLOT_ROOT:
#   ffs_trees/          — branching trees per IC
#   trajectory_physics/ — per-pathway physics panels
#   committor_fields/   — committor field maps + heatmaps
#   committor_maps/     — per-IC committor overlays
#   reactive_trajectories/ — ensemble pathway maps
#   ffs_ensemble/       — ensemble strip plots
#   ffs_reactive_single/   — single reactive trajectory maps
#   summary/            — rates, enhancement, commitment curve
# ============================================================

set -euo pipefail

APPS=/glade/work/schreck/repos/miles-tails/applications
FFS_SCRATCH=/glade/derecho/scratch/schreck/FFS
FFS_CONFIG=${FFS_SCRATCH}/ffs.yml
MODEL_CONFIG=${FFS_SCRATCH}/model.yml
RESULTS=${FFS_SCRATCH}/results_mar18
FFS_CSV=${RESULTS}/ffs_statistics_all_ics.csv
IFS_CSV=${RESULTS}/IFS/ifs_rates_FFS.csv
PLOT_ROOT=${RESULTS}/plots

ALL_ICS="2022-09-02T00Z 2022-09-04T00Z 2022-09-22T00Z"

# Create all subdirectories
mkdir -p \
    ${PLOT_ROOT}/ffs_trees \
    ${PLOT_ROOT}/trajectory_physics \
    ${PLOT_ROOT}/committor_fields \
    ${PLOT_ROOT}/committor_maps \
    ${PLOT_ROOT}/reactive_trajectories \
    ${PLOT_ROOT}/ffs_ensemble \
    ${PLOT_ROOT}/ffs_reactive_single \
    ${PLOT_ROOT}/summary

echo "=== results_mar18 figures → ${PLOT_ROOT} ==="

# ============================================================
# 1. Rates + interface probabilities  (all ICs)
# ============================================================
echo "=== [1] plot_rates_comparison.py ==="
python ${APPS}/plot_rates_comparison.py \
    --ffs_config ${FFS_CONFIG} \
    --ffs_csv    ${FFS_CSV} \
    --ifs_csv    ${IFS_CSV} \
    --plot_dir   ${PLOT_ROOT}/summary

# ============================================================
# 2. Commitment curve  (all ICs)
# ============================================================
echo "=== [2] plot_commitment_curve.py ==="
python ${APPS}/plot_commitment_curve.py \
    --ffs_config ${FFS_CONFIG} \
    --ffs_csv    ${FFS_CSV} \
    --ifs_csv    ${IFS_CSV} \
    --plot_dir   ${PLOT_ROOT}/summary

# ============================================================
# 3. Enhancement factor  (all ICs)
# ============================================================
echo "=== [3] plot_enhancement_factor.py ==="
python ${APPS}/plot_enhancement_factor.py \
    --ffs_config ${FFS_CONFIG} \
    --ffs_csv    ${FFS_CSV} \
    --plot_dir   ${PLOT_ROOT}/summary

# ============================================================
# 4. FFS trees  (all 3 ICs, top 5 with spread filter)
# ============================================================
echo "=== [4] plot_ffs_tree.py (all ICs) ==="
for ic in ${ALL_ICS}; do
    echo "  → ${ic}"
    python ${APPS}/plot_ffs_tree.py \
        --ffs_config  ${FFS_CONFIG} \
        --ic_dir      ${RESULTS}/${ic} \
        --plot_dir    ${PLOT_ROOT}/ffs_trees \
        --top_n       5 \
        --scan_n      30 \
        --min_spread  1.5 \
        --workers     16
done

# ============================================================
# 5. Reactive trajectories  (all 3 ICs)
# ============================================================
echo "=== [5] plot_reactive_trajectories.py (all ICs) ==="
for ic in ${ALL_ICS}; do
    ic_time="${ic/T/ }"   # 2022-09-02T00Z → 2022-09-02 00Z
    ic_time="${ic_time/00Z/00:00:00}"
    echo "  → ${ic}"
    python ${APPS}/plot_reactive_trajectories.py \
        --ffs_config ${FFS_CONFIG} \
        --ic_dir     ${RESULTS}/${ic} \
        --plot_dir   ${PLOT_ROOT}/reactive_trajectories \
        --workers    16
done

# ============================================================
# 6. FFS ensemble strip plots  (all 3 ICs)
# ============================================================
echo "=== [6] plot_ffs_ensemble.py (all ICs) ==="
for ic in ${ALL_ICS}; do
    echo "  → ${ic}"
    python ${APPS}/plot_ffs_ensemble.py \
        --ffs_config ${FFS_CONFIG} \
        --ic_dir     ${RESULTS}/${ic} \
        --plot_dir   ${PLOT_ROOT}/ffs_ensemble \
        --workers    16
done

# ============================================================
# 7. Committor maps  (all 3 ICs)
# ============================================================
echo "=== [7] plot_committor_map.py (all ICs) ==="
for ic in ${ALL_ICS}; do
    echo "  → ${ic}"
    python ${APPS}/plot_committor_map.py \
        --ffs_config ${FFS_CONFIG} \
        --ic_dir     ${RESULTS}/${ic} \
        --plot_dir   ${PLOT_ROOT}/committor_maps \
        --workers    16
done

# ============================================================
# 8. Committor fields  (all ICs aggregated)
# ============================================================
echo "=== [8] plot_committor_fields.py ==="
python ${APPS}/plot_committor_fields.py \
    --ffs_config ${FFS_CONFIG} \
    --ffs_csv    ${FFS_CSV} \
    --output_dir ${RESULTS} \
    --plot_dir   ${PLOT_ROOT}/committor_fields \
    --workers    16 \
    --min_samples 5 \
    --n_bins     15

# ============================================================
# 9. Bottleneck heatmap  (all ICs)
# ============================================================
echo "=== [9] plot_bottleneck_heatmap.py ==="
python ${APPS}/plot_bottleneck_heatmap.py \
    --ffs_config ${FFS_CONFIG} \
    --ffs_csv    ${FFS_CSV} \
    --plot_dir   ${PLOT_ROOT}/summary

# ============================================================
# 10. Trajectory physics  (all 3 ICs, top pathways)
#     Requires GPU — run on casper with model.yml present
# ============================================================
echo "=== [10] plot_trajectory_physics.py — list available pathways ==="
for ic in ${ALL_ICS}; do
    ic_time="${ic/T/ }"
    ic_time="${ic_time/00Z/00:00:00}"
    echo "  → ${ic}"
    python ${APPS}/plot_trajectory_physics.py \
        --model_config ${MODEL_CONFIG} \
        --ffs_config   ${FFS_CONFIG} \
        --ic_time      "${ic_time}" \
        --list_pathways
done

echo ""
echo "=== DONE ==="
echo "All figures in: ${PLOT_ROOT}"
echo ""
echo "Subdirectory summary:"
for d in ffs_trees trajectory_physics committor_fields committor_maps \
          reactive_trajectories ffs_ensemble ffs_reactive_single summary; do
    count=$(ls ${PLOT_ROOT}/${d}/*.png 2>/dev/null | wc -l)
    echo "  ${d}/  →  ${count} PNGs"
done
