#!/bin/bash
#SBATCH --job-name=lf_vanilla_rn50            # Job name
#SBATCH --output=outputs/lf_vanilla_rn50_%j.log  # Standard output log (%j = job ID)
#SBATCH --error=outputs/lf_vanilla_rn50_%j.err   # Standard error log
#SBATCH --time=2-00:00:00                     # Time limit (dd-hh:mm:ss)
#SBATCH --ntasks=2                            # Number of tasks (typically 1 for single-node jobs)
#SBATCH --cpus-per-task=8                     # Number of CPUs per task
#SBATCH --mem=48GB                            # Memory allocation
#SBATCH --partition=ada                       # Partition (long/queue)
#SBATCH --gres=gpu:ADA6000:2                  # GPU allocation (if needed, modify accordingly)
#SBATCH --account=research
#SBATCH --requeue                            # auto-requeue on node failure/preemption
#SBATCH --open-mode=append                   # append logs across requeues
# =============================================================
echo "job: $SLURM_JOB_NAME"
# >>> Conda setup <<<
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clip_ddetr

# Job execution commands
. ./.env
echo $COCO_HOME
echo $SLURM_JOBID

# 1) Find a free port by binding to port 0
export MASTER_PORT=$(python - <<'EOF'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 0))
port = s.getsockname()[1]
s.close()
print(port)
EOF
)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_NNODES=${SLURM_NNODES:-1}
export SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-1}
echo "nnodes: $SLURM_NNODES"
echo "nproc_per_node: $SLURM_GPUS_ON_NODE"
echo "master port: $MASTER_PORT"

# Auto-resume: pass --resume only if a checkpoint already exists (safe on first run + requeue)
OUT_DIR="$PROJECT_HOME/outputs/lf_vanilla_rn50"
mkdir -p "$OUT_DIR"
RESUME_ARG=""
if [ -f "$OUT_DIR/checkpoint.pth" ]; then
    RESUME_ARG="--resume $OUT_DIR/checkpoint.pth"
    echo "Resuming from $OUT_DIR/checkpoint.pth"
else
    echo "No checkpoint found; starting fresh."
fi

python -u -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --master_port $MASTER_PORT \
    main.py \
    --method vidt \
    --backbone_name swin_tiny \
    --epochs 12 \
    --lr 1e-4 \
    --min-lr 1e-7 \
    --batch_size 7 \
    --num_workers 14 \
    --aux_loss True \
    --with_box_refine True \
    --coco_path $COCO_HOME \
    --qd_root $QD_DATASET \
    --output_dir $OUT_DIR \
    --start_epoch 0 \
    --lr_drop 20 \
    --warmup-epochs 0 \
    $RESUME_ARG
