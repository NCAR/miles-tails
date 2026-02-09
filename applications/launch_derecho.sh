#!/bin/bash
#PBS -A NAML0001
#PBS -N hurricane_ffs
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -q main
#PBS -j oe
#PBS -k eod
##PBS -r n

# ============================================================================
# Environment Setup
# ============================================================================
module load ncarenv/24.12 gcc/12.4.0 ncarcompilers cray-mpich/8.1.29 \
            cuda/12.3.2 conda/latest cudnn/9.2.0.82-12 mkl/2025.0.1

conda activate /glade/work/schreck/conda-envs/torch28-nccl221

export PYTHONPATH=/glade/work/schreck/repos/miles-credit:${PYTHONPATH}
export PYTHONPATH=/glade/work/schreck/repos/miles-tails:${PYTHONPATH}
export LSCRATCH=/glade/derecho/scratch/schreck/
export LOGLEVEL=INFO

# ============================================================================
# NCCL/GPU Configuration
# ============================================================================
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export NCCL_IB_DISABLE=1
export NCCL_CROSS_NIC=1
export NCCL_NCHANNELS_PER_NET_PEER=4
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PBH

# ============================================================================
# MPICH/Libfabric Configuration
# ============================================================================
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_RDMA_ENABLED_CUDA=1
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_OPTIMIZED_MRS=false
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

# ============================================================================
# Job Configuration
# ============================================================================
SCRIPT_DIR=/glade/work/schreck/repos/miles-tails/applications
FFS_SCRIPT=${SCRIPT_DIR}/run_parallel_ffs.py
MODEL_CONFIG=model.yml
FFS_CONFIG=ffs.yml

# Set IC index (override with PBS_ARRAY_INDEX if using job arrays)
IC_INDEX=${PBS_ARRAY_INDEX:-0}

# ===== CONFIGURE PHASE HERE =====
PHASE=shoot      # Options: flux, shoot
INTERFACE=4      # Only used for shoot phase -- start at 0 to shoot from lambda_0 to lambda_1, ... N-1 interfaces
NUM_WORKERS=2

echo "============================================================================"
echo "Hurricane FFS Job Configuration"
echo "============================================================================"
echo "Phase:           ${PHASE}"
if [ "${PHASE}" = "shoot" ]; then
    echo "Interface:       ${INTERFACE}"
fi
echo "IC Index:        ${IC_INDEX}"
echo "Workers per GPU: ${NUM_WORKERS}"
echo "Total GPUs:      4"
echo "============================================================================"

# ============================================================================
# Launch 4 parallel processes (1 per GPU)
# ============================================================================
if [ "${PHASE}" = "flux" ]; then
    # FLUX GENERATION
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
    # SHOOTING PHASE
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

echo "============================================================================"
echo "All GPU processes completed"
echo "============================================================================"