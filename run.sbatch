#!/bin/bash
#SBATCH --job-name=qng_baseline
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

echo "Host: $(hostname)"
echo "JobID: $SLURM_JOB_ID"

cd /nfshomes/aysingha/workspace/quantum-manifold-optimization

source /fs/nexus-scratch/aysingha/conda/miniconda3/etc/profile.d/conda.sh
conda activate qng-baseline

python -u run_all_parallel.py --workers $SLURM_CPUS_PER_TASK
