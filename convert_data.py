"""
Usage: 
    (1) First run:
        `python -u convert_data.py ... --shard_size 10000`
    (2) Resume later: 
        `python -u convert_data.py ... --shard_size 10000 --resume`
    (3) Force rebuild:
        `python -u convert_data.py ... --shard_size 10000 --overwrite`
"""
import os, io, tarfile, json, argparse
from pathlib import Path
from PIL import Image
import pickle
from PIL import ImageOps
import multiprocessing as mp
from datasets.coco import convert_to_PIL
from dotenv import load_dotenv
load_dotenv()

COCO_HOME = Path(os.environ.get("COCO_HOME"))
PROJECT_HOME = Path(os.environ.get("PROJECT_HOME"))

def pkl_to_png_bytes(pkl_path: str, size: int = 224) -> bytes:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    key = next(iter(obj.keys()))
    drawing = obj[key]
    pil_img = convert_to_PIL(drawing)
    pil_img.thumbnail((size, size), getattr(Image, "Resampling", Image).LANCZOS)
    pil_img = pil_img.convert("L")
    pil_img = ImageOps.invert(pil_img)

    canvas = Image.new("L", (size, size), color=0)
    x0 = (size - pil_img.size[0]) // 2
    y0 = (size - pil_img.size[1]) // 2
    canvas.paste(pil_img, (x0, y0))
    
    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)  # good for sparse sketches
    return buf.getvalue()

def worker_init():
    # avoid MKL/OMP oversubscription per worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def process_one(args):
    # args = (i, pkl_path, label, size)
    i, pkl_path, label, size = args
    png_bytes = pkl_to_png_bytes(pkl_path, size=size)
    return i, png_bytes, label

def add_bytes_to_tar(tar: tarfile.TarFile, arcname: str, data: bytes):
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))
    
def _resolve_path(sketch_path):
    sketch_path = Path(sketch_path)
    if sketch_path.is_absolute():
        return Path(*COCO_HOME.parts, *sketch_path.parts[4:])
    return sketch_path
    

def index_generator(image_set):
    """
        Generates:
        [
            {"pkl": "/path/to/sample0.pkl", "label": 12},
            {"pkl": "/path/to/sample1.pkl", "label": 12}
        ]
    """
    _quickdraw_path = PROJECT_HOME / "checkpoints/processed_quick_draw_paths_purified.pkl"  # TODO: Refactor this
    _quickdraw_path = pickle.load(open(_quickdraw_path, 'rb'))
    if image_set == 'train':
        _quickdraw_path = _quickdraw_path['train_x']
    else:
        _quickdraw_path = _quickdraw_path['valid_x']
    index = []
    for path in _quickdraw_path:
        cat = path.split('/')[-2]
        index.append({"pkl": str(_resolve_path(path)), "label": cat})
    return index
def shard_path(out_dir: Path, prefix: str, shard_id: int) -> Path:
    return out_dir / f"{prefix}-{shard_id:05d}.tar"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_json", type=str, required=True,
                    help="JSON list of {'pkl': '/abs/or/rel/path.pkl', 'label': int}")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--prefix", type=str, default="train")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--shard_size", type=int, default=5000, help="samples per tar shard")
    # array-friendly shard range
    ap.add_argument("--shard_start", type=int, default=0, help="First shard id (inclusive)")
    ap.add_argument("--shard_end", type=int, default=-1, help="Last shard id (exclusive). -1 = until end")
    # multiprocessing
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--chunksize", type=int, default=32)
    # resume/overwrite
    ap.add_argument("--resume", action="store_true",
                help="Skip shards that already exist and resume from the first missing shard.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing shards (disables resume behavior).")
    ap.add_argument("--min_shard_bytes", type=int, default=1024 * 1024,
                    help="If an existing shard is smaller than this, treat it as incomplete and remake it.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(args.index_json):
        print(f"Generating index json for {args.prefix}...", flush=True)
        samples = index_generator(args.prefix)
        with open(args.index_json, "w") as f:
            json.dump(samples, f)
    else:
        samples = json.load(open(args.index_json, "r"))
    n = len(samples)
    print(f"Loaded {n} samples")
    
    total_shards = (n + args.shard_size - 1) // args.shard_size
    shard_start = max(0, args.shard_start)
    shard_end = total_shards if args.shard_end == -1 else min(args.shard_end, total_shards)

    if shard_start >= shard_end:
        print(f"Nothing to do: shard range [{shard_start}, {shard_end}) is empty.", flush=True)
        return
    
     # If resume without explicit shard range, find first missing shard globally
    if args.resume and args.shard_start == 0 and args.shard_end == -1 and not args.overwrite:
        sid = 0
        while sid < total_shards:
            sp = shard_path(out_dir, args.prefix, sid)
            if not sp.exists():
                break
            if sp.stat().st_size < args.min_shard_bytes:
                print(f"Found small/incomplete shard {sp} ({sp.stat().st_size} bytes). Will remake from here.", flush=True)
                break
            sid += 1
        shard_start = max(shard_start, sid)
        print(f"Resuming from shard {shard_start} (global scan).", flush=True)

    print(f"Shard plan: total_shards={total_shards}, processing [{shard_start}, {shard_end})", flush=True)

    ctx = mp.get_context("spawn") # safer with PIL on clusters
    pool = ctx.Pool(processes=args.num_workers, initializer=worker_init)
    
    try:
        for shard_id in range(shard_start, shard_end):
            sp = shard_path(out_dir, args.prefix, shard_id)

            if sp.exists() and not args.overwrite:
                if args.resume and sp.stat().st_size >= args.min_shard_bytes:
                    print(f"Skipping existing shard {sp}", flush=True)
                    continue
                elif args.resume and sp.stat().st_size < args.min_shard_bytes:
                    print(f"Remaking small/incomplete shard {sp} ({sp.stat().st_size} bytes)", flush=True)
                else:
                    raise FileExistsError(f"{sp} exists. Use --resume to skip or --overwrite to rebuild.")

            start_i = shard_id * args.shard_size
            end_i = min((shard_id + 1) * args.shard_size, n)
            if start_i >= n:
                break

            tmp = str(sp) + ".tmp"
            print(f"[shard {shard_id:05d}] Writing {tmp} for samples [{start_i}, {end_i})", flush=True)

            tasks = [(j, samples[j]["pkl"], samples[j]["label"], args.size) for j in range(start_i, end_i)]

            results_it = pool.imap_unordered(process_one, tasks, chunksize=args.chunksize)

            with tarfile.open(tmp, "w") as tar:
                done = 0
                for j, png_bytes, label in results_it:
                    key = f"{j:09d}"
                    add_bytes_to_tar(tar, f"{key}.png", png_bytes)
                    add_bytes_to_tar(tar, f"{key}.cls", str(label).encode("utf-8"))
                    done += 1
                    if done % 1000 == 0:
                        print(f"[shard {shard_id:05d}] {done}/{len(tasks)}", flush=True)

            os.replace(tmp, sp)  # atomic commit
            print(f"[shard {shard_id:05d}] Finished {sp}", flush=True)

    finally:
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()
