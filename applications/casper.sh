#!/bin/bash -l
#PBS -N ffs
#PBS -l select=1:ncpus=8:mpiprocs=1:mem=128GB:ngpus=1:gpu_type=a100_80gb
#PBS -l walltime=12:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod

source ~/.bashrc
conda activate credit-casper

SCRIPT_DIR=/glade/work/schreck/repos/miles-tails/applications
FFS_SCRIPT=${SCRIPT_DIR}/run_parallel_ffs.py
MODEL_CONFIG=model.yml
FFS_CONFIG=ffs.yml

# ===== CONFIGURE PHASE HERE =====
PHASE=flux      # Options: flux, shoot
INTERFACE=0      # Only used for shoot phase
IC_INDEX=${PBS_ARRAY_INDEX}       # Initial condition index
NUM_WORKERS=4

echo "============================================================================"
echo "Hurricane FFS Job Configuration"
echo "============================================================================"
echo "Phase:           ${PHASE}"
if [ "${PHASE}" = "shoot" ]; then
    echo "Interface:       ${INTERFACE}"
fi
echo "IC Index:        ${IC_INDEX}"
echo "Workers per GPU: ${NUM_WORKERS}"
echo "============================================================================"

if [ "${PHASE}" = "flux" ]; then
    CUDA_VISIBLE_DEVICES=0 \
    torchrun --nproc_per_node=1 --master-port=$((RANDOM % 10000 + 20000)) \
        ${FFS_SCRIPT} \
        --model_config ${MODEL_CONFIG} \
        --ffs_config ${FFS_CONFIG} \
        --phase flux \
        --ic_index ${IC_INDEX} \
        --num_workers ${NUM_WORKERS} &
else
    CUDA_VISIBLE_DEVICES=0 \
    torchrun --nproc_per_node=1 --master-port=$((RANDOM % 10000 + 20000)) \
        ${FFS_SCRIPT} \
        --model_config ${MODEL_CONFIG} \
        --ffs_config ${FFS_CONFIG} \
        --phase shoot \
        --interface ${INTERFACE} \
        --ic_index ${IC_INDEX} \
        --num_workers ${NUM_WORKERS} &
fi

wait
echo "============================================================================"
echo "All GPU processes completed"
echo "============================================================================"