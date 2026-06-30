#!/bin/bash
# LocFormer OW/CW eval-suite runner on rahul (single GPU). Usage: bash eval_suite_rahul.sh {open|closed}
# Evaluates outputs/lf_ow_rn50/checkpoint.pth: open -> 14-cat holdout (OW), closed -> all-cats (CW).
set -e
cd /home/rahul/arka/locformer-SGOL
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clip_ddetr
source ./.env
export WANDB_MODE=offline            # eval-only: no online wandb run
export CUDA_VISIBLE_DEVICES=0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # torch>=2.6: allow loading ckpt w/ argparse Namespace (bare torch.load in main.py)
SCHEME="${1:-open}"
CKPT=outputs/lf_ow_rn50/checkpoint.pth
OUT="outputs/lf_ow_rn50/eval_rahul/${SCHEME}"
mkdir -p "$OUT"
PORT=$(python - <<'PY'
import socket
s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)
echo ">>> eval scheme=$SCHEME ckpt=$CKPT port=$PORT $(date)"
python -u -m torch.distributed.run --nproc_per_node=1 --master_port "$PORT" \
  main.py \
  --method vidt --backbone_name swin_tiny \
  --with_box_refine True --aux_loss True --num_sketches 1 \
  --train_scheme_world "$SCHEME" \
  --coco_path "$COCO_HOME" --qd_root "$QD_DATASET" \
  --eval True --resume "$CKPT" --output_dir "$OUT"
