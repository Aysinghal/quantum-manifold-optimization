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
# TASKS / OPTS feed run_all_parallel.py's --tasks / --optimizers flags.
# Each accepts a comma-separated list of TASK_TIERS / OPT_TIERS keys OR
# individual task/optimizer names (or any mix); the runner takes the union.
# SEEDS is optional and overrides the default [0,1,2,3,4].
# Defaults run the original baseline + rigged tasks under the standard
# QNG sweep (no QNG_full). Override examples:
#   TASKS=qng_advantage   OPTS=flat_only       sbatch run.sbatch
#   TASKS=sanity          OPTS=qng_full_sweep  sbatch run.sbatch
#   TASKS=sanity          OPTS=full_qng_sweep  sbatch run.sbatch  # incl. QNG_full
#   TASKS=manifold_torus  OPTS=torus_concept   sbatch run.sbatch  # Phase 1 (null)
#   TASKS=manifold_sphere OPTS=sphere_concept  sbatch run.sbatch  # Phase 2
#   TASKS=vqe_stokes_overparam_long OPTS=torus_concept SEEDS=0 sbatch run.sbatch
#       (single-task, single-seed diagnostic-style run; produces theta plot)
# Phase 2 job count: 7 opts x 8 tasks x 5 seeds = 280 jobs.
TASKS="${TASKS:-sanity,qng_advantage}"
OPTS="${OPTS:-qng_full_sweep}"
echo "  Shots    : $SHOTS  (0 = analytic noiseless + adjoint; >0 = shot-noisy + parameter-shift)"
echo "  Tasks    : $TASKS  (tier names and/or individual task names)"
echo "  Opts     : $OPTS   (tier names and/or individual optimizer names)"
if [ -n "$SEEDS" ]; then
    echo "  Seeds    : $SEEDS  (overriding default [0,1,2,3,4])"
fi
if [ -n "$TITLE" ]; then
    echo "  Title    : $TITLE  (slugified into the run-folder name)"
fi
echo "======================================================================"

SEEDS_FLAG=""
if [ -n "$SEEDS" ]; then
    SEEDS_FLAG="--seeds $SEEDS"
fi

# TITLE may contain spaces, so we branch and quote it as a single argv
# entry rather than splatting it through an unquoted variable.
if [ -n "$TITLE" ]; then
    python -u run_all_parallel.py --workers $SLURM_CPUS_PER_TASK --shots $SHOTS \
        --tasks $TASKS --optimizers $OPTS $SEEDS_FLAG --title "$TITLE"
else
    python -u run_all_parallel.py --workers $SLURM_CPUS_PER_TASK --shots $SHOTS \
        --tasks $TASKS --optimizers $OPTS $SEEDS_FLAG
fi

echo "======================================================================"
echo "  End      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"
