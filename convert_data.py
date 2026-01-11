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
from datasets.coco import convert_to_PIL
from dotenv import load_dotenv
load_dotenv()

COCO_HOME = Path(os.environ.get("COCO_HOME"))
PROJECT_HOME = Path(os.environ.get("PROJECT_HOME"))

def pkl_to_pil_uint8_grayscale(pkl_path: str, size: int = 224) -> Image.Image:
    obj = pickle.load(open(pkl_path, "rb"))
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
    return canvas

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
"""
needs index_json:
[
  {"pkl": "/path/to/sample0.pkl", "label": 12},
  {"pkl": "/path/to/sample1.pkl", "label": 12}
]
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_json", type=str, required=True,
                    help="JSON list of {'pkl': '/abs/or/rel/path.pkl', 'label': int}")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--prefix", type=str, default="train")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--shard_size", type=int, default=5000, help="samples per tar shard")
    ap.add_argument("--resume", action="store_true",
                help="Skip shards that already exist and resume from the first missing shard.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing shards (disables resume behavior).")
    ap.add_argument("--min_shard_bytes", type=int, default=1024 * 1024,
                    help="If an existing shard is smaller than this, treat it as incomplete and remake it.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    def shard_path(out_dir, prefix, shard_id):
        return out_dir / f"{prefix}-{shard_id:05d}.tar"

    start_shard_id = 0

    if args.overwrite:
        start_shard_id = 0
    elif args.resume:
        # find first shard that does NOT exist (or is suspiciously small)
        shard_id = 0
        while True:
            sp = shard_path(out_dir, args.prefix, shard_id)
            if not sp.exists():
                break
            if sp.stat().st_size < args.min_shard_bytes:
                print(f"Found small/incomplete shard {sp} ({sp.stat().st_size} bytes). Will remake from here.", flush=True)
                break
            shard_id += 1
        start_shard_id = shard_id
        print(f"Resuming: skipping shards [0..{start_shard_id-1}] and starting from shard {start_shard_id}.", flush=True)
    else:
        start_shard_id = 0

    start_i = start_shard_id * args.shard_size

    samples = json.load(open(args.index_json, "r"))
    n = len(samples)
    print(f"Loaded {n} samples")
    
    if start_i >= n:
        print("All shards already processed. Nothing to do.", flush=True)
        return
    
    index = index_generator(args.prefix)
    with open(COCO_HOME / f"processed_quickdraw/{args.prefix}_index.json", "w") as f:
        json.dump(index, f)

    shard_id = start_shard_id
    i = start_i
    tar = None

    while i < len(samples):
        # open shard at boundary
        if i % args.shard_size == 0:
            if tar is not None:
                tar.close()
            sp = shard_path(out_dir, args.prefix, shard_id)

            if args.resume and sp.exists() and sp.stat().st_size >= args.min_shard_bytes:
                print(f"Skipping existing shard: {sp}", flush=True)
                shard_id += 1
                i = shard_id * args.shard_size
                continue

            if sp.exists() and not args.overwrite:
                raise FileExistsError(f"{sp} exists. Use --resume or --overwrite.")

            print(f"Opening shard: {sp}", flush=True)
            tar = tarfile.open(sp, "w")

        s = samples[i]
        key = f"{i:09d}"
        pkl_path = s["pkl"]
        label = int(s["label"])

        img = pkl_to_pil_uint8_grayscale(pkl_path, size=args.size)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()

        add_bytes_to_tar(tar, f"{key}.png", png_bytes)
        add_bytes_to_tar(tar, f"{key}.cls", str(label).encode("utf-8"))

        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1}/{len(samples)}", flush=True)

        i += 1

    if tar is not None:
        tar.close()

    print("Done.")

if __name__ == "__main__":
    main()
