#!/bin/bash
#SBATCH --job-name=rn50_qd_pt                  # Job name
#SBATCH --output=outputs/rn50_qd_pt_%j.log     # Standard output log (%j = job ID)
#SBATCH --error=outputs/rn50_qd_pt_%j.err      # Standard error log
#SBATCH --time=1-00:00:00                      # Time limit (dd-hh:mm:ss)
#SBATCH --ntasks=2                             # 2 tasks -> 2 GPUs
#SBATCH --cpus-per-task=8                      # Number of CPUs per task
#SBATCH --mem=80GB                             # Memory allocation (a100 partition caps job RAM at 84GB)
#SBATCH --partition=a100                       # A100 partition (node cn6)
#SBATCH --gres=gpu:A100:2                      # 2x A100 80GB
#SBATCH --account=research                     # research is permitted on a100 (AllowQos=ALL)
#SBATCH --requeue                              # auto-requeue on node failure/preemption
#SBATCH --open-mode=append                     # append logs across requeues
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

# Auto-resume: pass --resume-from latest only if a checkpoint already exists
# (safe on first run + requeue). Checkpoints live under <checkpoint-dir>/<run-name>.
RESUME_ARG=""
if [ -f "$PROJECT_HOME/checkpoints/r50-sgd/latest.pth" ]; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from checkpoints/r50-sgd/latest.pth"
else
    echo "No checkpoint found; starting fresh."
fi

python -u -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --master_port $MASTER_PORT \
    resnet_pretrainer.py \
    --batch-size 1400 \
    --epochs 10 \
    `# 10 epochs x ~2.4h = ~24h, fits one a100 wall window (no requeue). 42M imgs is ample.` \
    --peak-lr 0.05 \
    `# fixed peak LR — do NOT linear-scale for this fine-tuning task (1.094 was diverging).` \
    --warmup-epochs 5 \
    `# per-iteration warmup over 5 x 15095 = 75475 iters (smooth Goyal ramp); cosine over the rest.` \
    --grad-clip 1.0 \
    `# clip_grad_norm_ after unscale — catches the loss spikes (val loss 12/22 collapses).` \
    --ckpt-every-steps 600 \
    `# 600 = 1/25 epoch @ batch 1400 x 2 GPUs (15095 steps/epoch); recompute if batch/GPU count changes.` \
    --run-name "r50-sgd" \
    --wandb-project "resnet50-quickdraw-pt" \
    --ddp \
    $RESUME_ARG