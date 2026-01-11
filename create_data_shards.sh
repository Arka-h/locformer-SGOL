#!/bin/bash
#SBATCH -J qd_shards
#SBATCH -p long
#SBATCH -A research
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --array=0-55%2
#SBATCH -o logs/qd_shards_%A_%a.out
#SBATCH -e logs/qd_shards_%A_%a.err

set -euo pipefail
mkdir -p logs

# Avoid CPU oversubscription per process
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SHARD_SIZE=10000

# 5,600,000 samples / 10,000 per shard = 560 shards
TOTAL_SHARDS=560

# 56 array tasks × 10 shards each = 560 shards
SHARDS_PER_TASK=10
START=$(( SLURM_ARRAY_TASK_ID * SHARDS_PER_TASK ))
END=$(( START + SHARDS_PER_TASK ))
if [ "$END" -gt "$TOTAL_SHARDS" ]; then END="$TOTAL_SHARDS"; fi

echo "[$(date)] Host=$(hostname) Job=$SLURM_JOB_ID Task=$SLURM_ARRAY_TASK_ID Shards=[$START,$END)"
echo "INDEX_JSON=$INDEX_JSON"
echo "OUT_DIR=$OUT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate clip_ddetr
. ./.env
PYTHONUNBUFFERED=1 python convert_data.py --index_json $COCO_HOME/processed_quickdraw/train_index.json \
                            --out_dir $COCO_HOME/processed_quickdraw/tar_files/train_shards \
                            --prefix train \
                            --shard_size 10000 \
                            --shard_start "$START" \
                            --shard_end "$END" \
                            --resume