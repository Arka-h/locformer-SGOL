"""
Convert QuickDraw PKL strokes -> LMDB with key "{class}/{filename_stem}".

Usage examples:

(1) Convert ENTIRE dataset by scanning root:
python -u convert_quickdraw_pkl_to_lmdb.py \
  --qd_root "$COCO_HOME/quickdraw" \
  --out_lmdb "$COCO_HOME/processed_quickdraw/lmdb/quickdraw_all.lmdb" \
  --size 224 --num_workers 32 --map_size_gb 250

(2) Convert ONLY a filtered list of PKL paths (your seen-only list):
python -u convert_quickdraw_pkl_to_lmdb.py \
  --paths_pkl "$PROJECT_HOME/checkpoints/processed_quick_draw_paths_purified.pkl" \
  --paths_key train_x \
  --out_lmdb "$COCO_HOME/processed_quickdraw/lmdb/quickdraw_seen_train.lmdb" \
  --size 224 --num_workers 32 --map_size_gb 250

(3) Resume later:
... add --resume
"""

import argparse, io, os, pickle, struct
from pathlib import Path
import multiprocessing as mp

import lmdb
from PIL import Image, ImageOps

# your stroke->PIL function
from datasets.coco import convert_to_PIL

META_LEN_KEY = b"__len__"
META_CLASSES_KEY = b"__classes__"

def worker_init():
    # avoid CPU thread oversubscription per worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

def encode_value(label_str: str, png_bytes: bytes) -> bytes:
    # [uint16 label_len][label_bytes][png_bytes]
    lb = label_str.encode("utf-8")
    if len(lb) > 65535:
        raise ValueError("Label too long")
    return struct.pack("<H", len(lb)) + lb + png_bytes

def pkl_to_png_bytes(pkl_path: str, size: int = 224) -> bytes:
    obj = pickle.load(open(pkl_path, "rb"))
    key = next(iter(obj.keys()))
    drawing = obj[key]

    pil_img = convert_to_PIL(drawing)
    pil_img.thumbnail((size, size), getattr(Image, "Resampling", Image).LANCZOS)
    pil_img = pil_img.convert("L")
    pil_img = ImageOps.invert(pil_img)

    # center on black canvas
    canvas = Image.new("L", (size, size), color=0)
    x0 = (size - pil_img.size[0]) // 2
    y0 = (size - pil_img.size[1]) // 2
    canvas.paste(pil_img, (x0, y0))

    buf = io.BytesIO()
    # lossless + compact for sparse sketches
    canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def make_key_from_path(p: Path) -> tuple[str, str]:
    """
    key = "{class}/{stem}"
    label = class
    Assumes directory structure: .../<class>/<filename>.pkl
    """
    cls = p.parent.name
    stem = p.stem
    return f"{cls}/{stem}", cls

def iter_all_pkls(root: Path):
    # scans <root>/**/**/*.pkl
    for p in root.rglob("*.pkl"):
        if p.is_file():
            yield p

def process_one(args):
    pkl_path, size = args
    p = Path(pkl_path)
    k, label = make_key_from_path(p)
    png = pkl_to_png_bytes(str(p), size=size)
    return k, label, png

def build_lmdb_from_paths(
    paths,
    out_lmdb: Path,
    size: int,
    num_workers: int,
    map_size_gb: int,
    commit_every: int,
    resume: bool,
):
    out_lmdb.parent.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(out_lmdb),
        map_size=int(map_size_gb * (1024**3)),
        subdir=False,
        readonly=False,
        lock=not resume,   # resume mode sometimes used on shared FS
        readahead=False,
        meminit=False,
        max_readers=2048,
    )

    # resume counter
    with env.begin(write=False) as txn:
        n0 = txn.get(META_LEN_KEY)
        total = int(n0.decode("utf-8")) if n0 else 0

    classes_seen = set()
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=num_workers, initializer=worker_init)

    txn = env.begin(write=True)
    try:
        # stream conversion results; no huge RAM
        it = pool.imap_unordered(
            process_one,
            ((str(p), size) for p in paths),
            chunksize=64,
        )

        for k_str, label_str, png_bytes in it:
            k = k_str.encode("utf-8")

            if resume and txn.get(k) is not None:
                continue

            txn.put(k, encode_value(label_str, png_bytes), overwrite=not resume)
            total += 1
            classes_seen.add(label_str)

            if total % commit_every == 0:
                txn.put(META_LEN_KEY, str(total).encode("utf-8"), overwrite=True)
                txn.commit()
                txn = env.begin(write=True)
                print(f"[LMDB] wrote {total} samples...", flush=True)

        # final commit
        txn.put(META_LEN_KEY, str(total).encode("utf-8"), overwrite=True)
        # store classes (small) for convenience
        cls_blob = "\n".join(sorted(classes_seen)).encode("utf-8")
        txn.put(META_CLASSES_KEY, cls_blob, overwrite=True)
        txn.commit()

        print(f"[LMDB] DONE. total={total} output={out_lmdb}", flush=True)

    finally:
        pool.close()
        pool.join()
        env.sync()
        env.close()

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--qd_root", type=str, help="Root dir containing class folders with .pkl files")
    g.add_argument("--paths_pkl", type=str, help="Pickle file containing dict or list of pkl paths (your filtered list)")

    ap.add_argument("--paths_key", type=str, default=None,
                    help="If --paths_pkl is a dict, use this key (e.g. train_x / valid_x).")
    ap.add_argument("--out_lmdb", type=str, required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--map_size_gb", type=int, default=250)
    ap.add_argument("--commit_every", type=int, default=20000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.qd_root:
        root = Path(args.qd_root)
        paths = list(iter_all_pkls(root))
        print(f"[LMDB] scanning root: found {len(paths)} pkls", flush=True)
    else:
        obj = pickle.load(open(args.paths_pkl, "rb"))
        if isinstance(obj, dict):
            if args.paths_key is None:
                raise ValueError("--paths_key required when --paths_pkl is a dict")
            paths = [Path(p) for p in obj[args.paths_key]]
        else:
            paths = [Path(p) for p in obj]
        print(f"[LMDB] loaded list: {len(paths)} pkls", flush=True)

    build_lmdb_from_paths(
        paths=paths,
        out_lmdb=Path(args.out_lmdb),
        size=args.size,
        num_workers=args.num_workers,
        map_size_gb=args.map_size_gb,
        commit_every=args.commit_every,
        resume=args.resume,
    )

if __name__ == "__main__":
    main()
