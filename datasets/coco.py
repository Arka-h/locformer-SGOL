# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO detection dataset paired with QuickDraw sketch queries.

Sketch loading follows the mmap-backed approach from SLIP/datasets.py
(ptr + strokes .npy files, rasterised on-the-fly with rasterize_stroke3).
"""
from hashlib import new
import json
from pathlib import Path
import torch
import torch.utils.data
import torchvision

from pycocotools import mask as coco_mask
import datasets.transforms as T
import pickle
from collections import defaultdict
import random
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import time
import os


# ---------------------------------------------------------------------------
# Stroke rasteriser (mirrors SLIP/datasets.py rasterize_stroke3)
# ---------------------------------------------------------------------------
def rasterize_stroke3(strokes, size=224, line_width=2, padding=10):
    """
    Convert stroke-3 format (T, 3) int16 to a white-background PIL RGB image.

    pen_state=0: pen down — continue current stroke
    pen_state=1: pen up   — end this stroke, next point starts a fresh one
    """
    abs_coords = np.cumsum(strokes[:, :2], axis=0).astype(float)
    pen_states  = strokes[:, 2]

    x, y = abs_coords[:, 0], abs_coords[:, 1]
    x_range = x.max() - x.min() or 1
    y_range = y.max() - y.min() or 1
    scale = (size - 2 * padding) / max(x_range, y_range)
    x = ((x - x.min()) * scale + padding).astype(int)
    y = ((y - y.min()) * scale + padding).astype(int)

    img  = Image.new('RGB', (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    stroke_points = []
    for i in range(len(strokes)):
        stroke_points.append((int(x[i]), int(y[i])))
        if pen_states[i] == 1:
            if len(stroke_points) >= 2:
                draw.line(stroke_points, fill=(0, 0, 0), width=line_width)
            stroke_points = []

    if len(stroke_points) >= 2:
        draw.line(stroke_points, fill=(0, 0, 0), width=line_width)

    return img


# ---------------------------------------------------------------------------
# Legacy vector-drawing helpers (kept for convert_data.py compatibility)
# ---------------------------------------------------------------------------
def convert_to_PIL(drawing, width=224, height=224):
    pil_img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(pil_img)
    for x, y in drawing:
        for i in range(1, len(x)):
            draw.line((x[i-1], y[i-1], x[i], y[i]), fill=0)
    return pil_img


def normalize_transform():
    return torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


# ---------------------------------------------------------------------------
# QuickDraw mmap index — mirrors SLIP's QuickDraw base class
# ---------------------------------------------------------------------------
class QuickDrawIndex:
    """
    Mmap-backed per-class index over QuickDraw SketchRNN npy files.

    Mirrors SLIP's QuickDraw base class.  Classes are loaded in parallel via
    ThreadPoolExecutor (avoids 10-20 min NFS startup delays).  Per-class
    sample index arrays are numpy int32 (not Python lists) so they are
    shared read-only across DataLoader workers after fork without CoW copies.

    Expected files under `root`:
        {class_name}.{split}.ptr.npy      — shape (N+1,) int64 byte offsets
        {class_name}.{split}.strokes.npy  — shape (total_strokes, 3) int16

    `split` is 'train', 'valid', or 'test'.
    """

    def __init__(self, root: str, class_names: list, split: str = 'train'):
        assert split in ('train', 'valid', 'test'), \
            f"split must be 'train', 'valid', or 'test', got {split!r}"
        self.root        = root
        self.split       = split
        self.class_names = list(class_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        # class_name -> (ptr ndarray, strokes mmap)
        self._mmap_cache: dict = {}
        # class_name -> np.arange(n, dtype=np.int32)  — fork-safe, no CoW
        self.class2indices: dict = {}

        def _load_one(class_name):
            ptr_path     = os.path.join(root, f'{class_name}.{split}.ptr.npy')
            strokes_path = os.path.join(root, f'{class_name}.{split}.strokes.npy')
            if not (os.path.exists(ptr_path) and os.path.exists(strokes_path)):
                return class_name, None, None
            ptr     = np.load(ptr_path)
            strokes = np.load(strokes_path, mmap_mode='r')
            return class_name, ptr, strokes

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=32) as ex:
            results = list(ex.map(_load_one, self.class_names))

        missing = []
        for class_name, ptr, strokes in results:
            if ptr is None:
                missing.append(class_name)
                continue
            self._mmap_cache[class_name]   = (ptr, strokes)
            self.class2indices[class_name] = np.arange(len(ptr) - 1, dtype=np.int32)

        if missing:
            print(f'[QuickDrawIndex] Warning: npy files not found for '
                  f'{len(missing)} class(es): {missing[:5]}')

    def has_class(self, class_name: str) -> bool:
        return class_name in self._mmap_cache

    def sample_pil(self, class_name: str, k: int = 5) -> list:
        """Return k rasterised PIL images sampled randomly from `class_name`."""
        ptr, strokes = self._mmap_cache[class_name]
        indices = np.random.choice(self.class2indices[class_name], size=k, replace=True)
        return [rasterize_stroke3(strokes[ptr[i]: ptr[i + 1]]) for i in indices]


# ---------------------------------------------------------------------------
# Dataset: COCO detection + QuickDraw sketch queries
# ---------------------------------------------------------------------------
class CocoDetectionQD(torchvision.datasets.CocoDetection):
    """
    COCO detection dataset where each sample is paired with k QuickDraw
    sketches of the *query* object category, loaded via SLIP-style mmap npy
    files (rasterised on-the-fly with rasterize_stroke3).

    Returns: (image_tensor, target_dict, sketch_tensor)
        sketch_tensor: float32 [k, 3, 224, 224], normalised
    """
    # UNSEEN_CATS follows Locformer's i%4==0 of their category ordering, not clip_ddetr's. 
    # !Do not change to match clip_ddetr without also re-pretraining
    # Categories visible during training (matches original subset)
    ALL_CATEGORIES = [
        'elephant', 'bear', 'cat', 'zebra', 'bus', 'horse', 'giraffe',
        'airplane', 'bed', 'dog', 'scissors', 'train', 'sandwich', 'pizza',
        'cow', 'broccoli', 'umbrella', 'sheep', 'bird', 'stop sign',
        'toothbrush', 'bicycle', 'hot dog', 'laptop', 'toaster', 'microwave',
        'banana', 'baseball bat', 'donut', 'couch', 'keyboard', 'cake',
        'oven', 'carrot', 'bench', 'suitcase', 'fire hydrant', 'fork',
        'chair', 'wine glass', 'apple', 'truck', 'cell phone', 'cup', 'car',
        'knife', 'toilet', 'clock', 'backpack', 'spoon', 'vase', 'book',
        'skateboard', 'sink', 'mouse', 'traffic light',
    ]

    def __init__(self, image_set, img_folder, ann_file, root, qd_root,
                 transforms, return_masks, num_sketches=5):
        json_file = json.load(open(ann_file))
        self.coco_home = Path(root)
        ROOT = self.coco_home / 'annotations'

        self.id2class = {}
        self.class2id = {}
        for cat in json_file['categories']:
            self.id2class[cat['id']] = cat['name']
            self.class2id[cat['name']] = cat['id']

        # Filter annotations to the supported category subset
        annotate, selected_image_ids = [], []
        for anno in json_file['annotations']:
            if self.id2class[anno['category_id']] in self.ALL_CATEGORIES:
                annotate.append(anno)
                selected_image_ids.append(anno['image_id'])

        selected_image_ids = set(selected_image_ids)
        images = [img for img in json_file['images']
                  if img['id'] in selected_image_ids]

        json_file['annotations'] = annotate
        json_file['images'] = images
        temp_ann = ROOT / f'temp_json_file_{image_set}.json'
        json.dump(json_file, open(temp_ann, 'w'))

        super().__init__(img_folder, temp_ann)
        self.image_set   = image_set
        self._transforms = transforms
        self.prepare     = ConvertCocoPolysToMask(return_masks)
        self.num_sketches = num_sketches

        self.transforms_sketch = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            normalize_transform(),
        ])

        # SLIP-style mmap index
        split = 'train' if image_set == 'train' else 'valid'
        print(f'[CocoDetectionQD] Loading QuickDraw mmap index ({split}) from {qd_root} ...')
        self.qd_index = QuickDrawIndex(qd_root, self.ALL_CATEGORIES, split=split)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        # Fix random seed for reproducible val sketches
        if self.image_set != 'train':
            random.seed(14)
        else:
            random.seed(int(1000 * time.time()) & 0xFFFFFFFF)

        image_id = self.ids[idx]
        target   = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # Pick a random category present in this image as the query
        categories = list(set(target['labels'].tolist()))
        if not categories:
            return self.__getitem__(random.randint(0, len(self) - 1))
        selected_cat = random.choice(categories)

        # Keep only annotations for the selected category
        keep = target['labels'] == selected_cat
        new_target = {}
        selected_keys = ['boxes', 'labels', 'area', 'iscrowd', 'masks']
        for key, value in target.items():
            new_target[key] = value[keep] if key in selected_keys else value
        new_target['labels'] = torch.ones_like(new_target['labels'])

        selected_cat_name = self.id2class[selected_cat]

        # Sample QuickDraw sketches for the query category via mmap
        if self.qd_index.has_class(selected_cat_name):
            pil_sketches = self.qd_index.sample_pil(selected_cat_name, k=self.num_sketches)
        else:
            # Fallback: blank white sketches if class is missing from the index
            pil_sketches = [Image.new('RGB', (224, 224), (255, 255, 255))] * self.num_sketches

        sketch_list = torch.stack([self.transforms_sketch(sk) for sk in pil_sketches])
        # sketch_list: [num_sketches, 3, 224, 224]

        old_boxes = new_target['boxes'].clone()
        if self._transforms is not None:
            img, new_target = self._transforms(img, new_target)

        if self.image_set != 'train':
            new_target['new_boxes'] = new_target['boxes']
            new_target['boxes']     = old_boxes

        return img, new_target, sketch_list


# ---------------------------------------------------------------------------
# Dataset: COCO detection + Sketchy sketch queries
# ---------------------------------------------------------------------------
class CocoDetectionSketchy(torchvision.datasets.CocoDetection):

    ALL_CATEGORIES = [
        'elephant', 'bear', 'cat', 'zebra', 'horse', 'giraffe', 'airplane',
        'dog', 'scissors', 'pizza', 'cow', 'umbrella', 'sheep', 'bicycle',
        'hot dog', 'banana', 'couch', 'bench', 'chair', 'apple', 'cup',
        'car', 'knife', 'clock', 'spoon', 'mouse', 'motorcycle',
    ]

    def __init__(self, image_set, img_folder, ann_file, root, sketchy_pkl,
                 sketchy_root, transforms, return_masks, num_sketches=5):
        json_file = json.load(open(ann_file))
        self.coco_home = Path(root)
        ROOT = self.coco_home / 'annotations'

        sketchy_data = pickle.load(open(sketchy_pkl, 'rb'))

        self.id2class = {}
        self.class2id = {}
        for cat in json_file['categories']:
            self.id2class[cat['id']] = cat['name']
            self.class2id[cat['name']] = cat['id']

        annotate, selected_image_ids = [], []
        for anno in json_file['annotations']:
            if self.id2class[anno['category_id']] in self.ALL_CATEGORIES:
                annotate.append(anno)
                selected_image_ids.append(anno['image_id'])

        selected_image_ids = set(selected_image_ids)
        images = [img for img in json_file['images']
                  if img['id'] in selected_image_ids]

        json_file['annotations'] = annotate
        json_file['images'] = images
        temp_ann = os.path.join(ROOT, f'temp_json_file_{image_set}.json')
        json.dump(json_file, open(temp_ann, 'w'))

        super().__init__(img_folder, temp_ann)
        self.image_set    = image_set
        self._transforms  = transforms
        self.prepare      = ConvertCocoPolysToMask(return_masks)
        self.num_sketches = num_sketches
        self.sketchy_root = sketchy_root

        self.transforms_sketch = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            normalize_transform(),
        ])

        self.class2quick = (sketchy_data['train'] if image_set == 'train'
                            else sketchy_data['valid'])
        print(f'[CocoDetectionSketchy] Loaded Sketchy ({image_set}).')

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        if self.image_set != 'train':
            random.seed(14)
        else:
            random.seed(int(1000 * time.time()) & 0xFFFFFFFF)

        image_id = self.ids[idx]
        target   = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        categories = list(set(target['labels'].tolist()))
        if not categories:
            return self.__getitem__(random.randint(0, len(self) - 1))
        selected_cat = random.choice(categories)

        keep = target['labels'] == selected_cat
        new_target = {}
        selected_keys = ['boxes', 'labels', 'area', 'iscrowd', 'masks']
        for key, value in target.items():
            new_target[key] = value[keep] if key in selected_keys else value
        new_target['labels'] = torch.ones_like(new_target['labels'])

        selected_cat_name = self.id2class[selected_cat]

        sketch_stems = random.choices(self.class2quick[selected_cat_name], k=self.num_sketches)
        sketch_list  = []
        for stem in sketch_stems:
            sk = Image.open(os.path.join(self.sketchy_root, stem + '.png')).convert('RGB')
            sketch_list.append(self.transforms_sketch(sk))
        sketch_list = torch.stack(sketch_list)   # [num_sketches, 3, 224, 224]

        old_boxes = new_target['boxes'].clone()
        if self._transforms is not None:
            img, new_target = self._transforms(img, new_target)

        if self.image_set != 'train':
            new_target['new_boxes'] = new_target['boxes']
            new_target['boxes']     = old_boxes

        return img, new_target, sketch_list


# ---------------------------------------------------------------------------
# COCO polygon / mask helpers
# ---------------------------------------------------------------------------
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = torch.tensor([target["image_id"]])
        anno     = [obj for obj in target["annotations"]
                    if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = torch.as_tensor(
            [obj["bbox"] for obj in anno], dtype=torch.float32
        ).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = torch.tensor([obj["category_id"] for obj in anno], dtype=torch.int64)

        if self.return_masks:
            masks = convert_coco_poly_to_mask(
                [obj["segmentation"] for obj in anno], h, w
            )

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = torch.as_tensor(
                [obj["keypoints"] for obj in anno], dtype=torch.float32
            )
            if keypoints.shape[0]:
                keypoints = keypoints.view(keypoints.shape[0], -1, 3)

        keep    = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes   = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {
            "boxes":     boxes,
            "labels":    classes,
            "image_id":  image_id,
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size":      torch.as_tensor([int(h), int(w)]),
            "area":      torch.tensor([obj["area"] for obj in anno])[keep],
            "iscrowd":   torch.tensor(
                [obj.get("iscrowd", 0) for obj in anno])[keep],
        }
        if self.return_masks:
            target["masks"] = masks
        if keypoints is not None:
            target["keypoints"] = keypoints

        return image, target


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------
def make_coco_transforms(image_set, args):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    scales = [480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640,
              656, 672, 688, 704, 720, 736, 752, 768, 784, 800]

    print(f'Resolution: shortest at most {max(scales)}')
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=scales[-1] * 1333 // 800),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=scales[-1] * 1333 // 800),
                ]),
            ),
            normalize,
        ])

    if image_set == 'val':
        print(args.eval_size)
        return T.Compose([
            T.RandomResize([args.eval_size], max_size=args.eval_size * 1333 // 800),
            normalize,
        ])

    raise ValueError(f'unknown image_set: {image_set}')


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    mode  = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val":   (root / "val2017",   root / "annotations" / f'{mode}_val2017.json'),
    }
    img_folder, ann_file = PATHS[image_set]

    return CocoDetectionQD(
        image_set, img_folder, ann_file, root,
        qd_root=args.qd_root,
        transforms=make_coco_transforms(image_set, args),
        return_masks=True,
    )
