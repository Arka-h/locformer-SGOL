source ~/miniconda3/etc/profile.d/conda.sh
conda activate clip_ddetr
. ./.env
PYTHONUNBUFFERED=1 python convert_data.py --index_json $COCO_HOME/processed_quickdraw/val_index.json \
                            --out_dir $COCO_HOME/processed_quickdraw/tar_files/val_shards \
                            --prefix val \
                            --shard_size 10000