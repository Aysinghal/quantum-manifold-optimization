#!/bin/bash
#SBATCH --job-name=qng_tier01
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=96
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=cbcb26,vulcan46,cml35,quics00
#SBATCH --requeue
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
# SHOTS=0 (analytic) is the default; the rigged qng_advantage tasks
# (vqe_stokes, vqe_heis_ring) are designed to demonstrate QNG's noise-free
# advantage, so a noisy default would muddy a negative result.
# Override with `SHOTS=1000 sbatch run.sbatch` for a hardware-realistic run.
SHOTS="${SHOTS:-0}"
# TASK_TIER selects which task tiers from run_all_parallel.TASK_TIERS run.
# OPT_TIER selects which optimizer tiers from run_all_parallel.OPT_TIERS run.
# Both accept comma-separated lists; the runner takes the union.
# Defaults run the original baseline + rigged tasks under the standard
# QNG sweep (no QNG_full). Override examples:
#   TASK_TIER=qng_advantage   OPT_TIER=flat_only       sbatch run.sbatch
#   TASK_TIER=sanity          OPT_TIER=qng_full_sweep  sbatch run.sbatch
#   TASK_TIER=sanity          OPT_TIER=full_qng_sweep  sbatch run.sbatch  # incl. QNG_full
#   TASK_TIER=manifold_torus  OPT_TIER=torus_concept   sbatch run.sbatch  # Phase 1 (null)
#   TASK_TIER=manifold_sphere OPT_TIER=sphere_concept  sbatch run.sbatch  # Phase 2 (this run)
# Phase 2 job count: 7 opts x 8 tasks x 5 seeds = 280 jobs.
TASK_TIER="${TASK_TIER:-sanity,qng_advantage}"
OPT_TIER="${OPT_TIER:-qng_full_sweep}"
echo "  Shots    : $SHOTS  (0 = analytic noiseless + adjoint; >0 = shot-noisy + parameter-shift)"
echo "  TaskTier : $TASK_TIER  (comma-separated TASK_TIERS keys)"
echo "  OptTier  : $OPT_TIER   (comma-separated OPT_TIERS keys)"
echo "======================================================================"

python -u run_all_parallel.py --workers $SLURM_CPUS_PER_TASK --shots $SHOTS \
    --task-tier $TASK_TIER --opt-tier $OPT_TIER

echo "======================================================================"
echo "  End      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"
