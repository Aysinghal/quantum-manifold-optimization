#!/bin/bash
#SBATCH --job-name=qng_baseline
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=80
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "======================================================================"
echo "  Host     : $(hostname)"
echo "  JobID    : $SLURM_JOB_ID"
echo "  CPUs     : $SLURM_CPUS_PER_TASK"
echo "  Workers  : $SLURM_CPUS_PER_TASK"
echo "  Mem      : $SLURM_MEM_PER_NODE MB"
echo "  Partition: $SLURM_JOB_PARTITION"
echo "  Start    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

cd /nfshomes/aysingha/workspace/quantum-manifold-optimization

source /fs/nexus-scratch/aysingha/conda/miniconda3/etc/profile.d/conda.sh
conda activate qng-baseline

echo "  Python   : $(which python) ($(python --version 2>&1))"
echo "  PennyLane: $(python -c 'import pennylane; print(pennylane.__version__)')"
echo "======================================================================"

python -u run_all_parallel.py --workers $SLURM_CPUS_PER_TASK

echo "======================================================================"
echo "  End      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"
