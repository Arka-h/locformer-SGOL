#!/bin/bash
#SBATCH --job-name=lf_qd_rn50                 # Job name
#SBATCH --output=outputs/lf_qd_rn50_%j.log    # Standard output log (%j = job ID)
#SBATCH --error=outputs/lf_qd_rn50_%j.err     # Standard error log
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
# Identical hyperparameters to train.sh (vanilla), EXCEPT the sketch encoder is
# initialized from the QuickDraw-pretrained ResNet50 (checkpoints/r50-sgd/best.pth,
# val acc 0.8507) via --sketch_encoder_pt. This is the QD-encoder comparison variant.
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
OUT_DIR="$PROJECT_HOME/outputs/lf_qd_rn50"
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
    --epochs 50 \
    --lr 1e-4 \
    `# head 1e-4; backbone 1e-5 (lr_backbone) and deformable proj 1e-5 (lr_linear_proj_mult) wired in build_optimizer` \
    --min-lr 1e-7 \
    --batch_size 5 \
    `# 5 per-GPU x 2 GPUs = effective batch 10` \
    --ckpt_every_steps 1000 \
    `# ~1/10 epoch @ batch 5 x 2 GPUs (9897 steps/epoch); modest IO for the 939MB checkpoint on a throttled FS` \
    --num_workers 14 \
    --aux_loss True \
    --with_box_refine True \
    --coco_path $COCO_HOME \
    --qd_root $QD_DATASET \
    --output_dir $OUT_DIR \
    --start_epoch 0 \
    --lr_drop 40 \
    `# single x0.1 drop at epoch 40 (~80% of 50); honored via multistep -> milestones=[lr_drop]` \
    --warmup-epochs 0 \
    --sketch_encoder_pt "$PROJECT_HOME/checkpoints/r50-sgd/best.pth" \
    `# QD-pretrained ResNet50 sketch encoder (val acc 0.8507) — the only difference vs train.sh` \
    $RESUME_ARG
