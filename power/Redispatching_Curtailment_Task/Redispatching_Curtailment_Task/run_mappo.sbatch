#!/bin/bash
#SBATCH --job-name=mappo-g2op
#SBATCH --array=0-2                 # one job per seed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64          # <- collection scales with this
#SBATCH --mem=180G                  # ~2-3 GB per collector worker
#SBATCH --time=12:00:00
#SBATCH --output=logs/mappo_seed%a_%A.out

# --- placeholders: set these to your cluster's values ---
# #SBATCH --partition=<partition>
# #SBATCH --account=<account>

set -euo pipefail
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate marl2grid_py312

# One thread per collector worker. Without this, every forked env process spawns
# $SLURM_CPUS_PER_TASK BLAS threads and they thrash.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GRID2OP_DATA_PATH=/scratch/dbst/phd/bsz178495/data_grid2op

python main.py \
    --n_frames 1_000_000 \
    --alg MAPPO \
    --lr 3e-5 \
    --MAPPO_n_episode 30 \
    --seeds "$SLURM_ARRAY_TASK_ID" \
    --n_envs auto
