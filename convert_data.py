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
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = json.load(open(args.index_json, "r"))
    n = len(samples)
    print(f"Loaded {n} samples")

    shard_id = -1
    tar = None

    index = index_generator(args.prefix)
    with open(COCO_HOME / f"processed_quickdraw/{args.prefix}_index.json", "w") as f:
        json.dump(index, f)
    
    for i, s in enumerate(samples):
        if i % args.shard_size == 0:
            if tar is not None:
                tar.close()
            shard_id += 1
            shard_path = out_dir / f"{args.prefix}-{shard_id:05d}.tar"
            print(f"Opening shard: {shard_path}")
            tar = tarfile.open(shard_path, "w")

        key = f"{i:09d}"
        pkl_path = s["pkl"]
        label = int(s["label"])

        img = pkl_to_pil_uint8_grayscale(pkl_path, size=args.size)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True) # PNG compresses sparse sketches very well
        png_bytes = buf.getvalue()

        add_bytes_to_tar(tar, f"{key}.png", png_bytes)
        add_bytes_to_tar(tar, f"{key}.cls", str(label).encode("utf-8"))

        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1}/{n}")

    if tar is not None:
        tar.close()
    print("Done.")

if __name__ == "__main__":
    main()
