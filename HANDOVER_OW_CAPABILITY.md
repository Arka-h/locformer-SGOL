# LocFormer Open-World (OW) Training — Handover Document

**Date:** 2026-06-27  
**Context:** Porting OW training capability from clip_ddetr to LocFormer to enable evaluation on held-out COCO categories using sketch-query conditioning.

---

## Executive Summary

**Goal:** Enable LocFormer to train and evaluate in an open-world setting where 14 COCO categories are held out during training and used as a held-out test set for evaluation.

**Key Finding:** LocFormer already has the *architectural capability* to do OW training — it features a binary match head (`num_classes=2`) and sketch-conditioned query processing, which is exactly what the paper's "Ours" open-world results (Table 2: mAP 12.2) rely on. **No architectural changes needed.** The missing piece is purely **data-side**: partitioning COCO into seen/unseen splits and filtering annotations at load time.

**Scope:** ~30 lines of changes in `datasets/coco.py`, no model/criterion/engine changes.

**Effort:** 4–6 hours (implementation + testing).

---

## Background: Why OW Training?

clip_ddetr demonstrated strong open-world detection results on COCO by:
1. **Holding out 14 categories** during training (unseen-cat split via `i%4==0` on 56 COCO categories).
2. **Training only on the 42 seen categories**.
3. **Evaluating per-epoch on the 14 unseen categories** to measure generalization to novel classes.

LocFormer's paper (Table 2, "Results in open-world, one-shot setting") reports similar OW results (**mAP 12.2**, outperforming baselines by 4.7%). Those results come from LocFormer's own architecture, not borrowed machinery. Reproducing them in the codebase requires implementing the data partitioning logic that the paper used.

---

## Analysis: Why LocFormer Can Do OW Without Architectural Changes

### Current Architecture

**LocFormer already has the two critical pieces:**

1. **Binary match head** (`methods/vidt/detector.py:567`):
   ```python
   num_classes = 2  # match / no-match, NOT 80-class linear
   class_embed_v2 = nn.Linear(hidden_dim*2, 2)
   ```
   — Unlike a standard 80-class COCO detector, LocFormer's head doesn't assume a fixed set of class indices. It scores queries as "match" or "no-match" against a *conditioned* target.

2. **Sketch-conditioned forward** (`methods/vidt/detector.py:373`):
   ```python
   def forward(self, samples, sketches):
       sketch_embedding = self.sketch_encoder(sketches)  # resnet50
       # ... query fusion ...
       return predictions_on_conditioned_queries
   ```
   — The model takes a sketch query as input. At eval, for an unseen category, you feed its QuickDraw sketch, and the model predicts whether image regions match that sketch. This is category-agnostic: the category knowledge is *in the sketch itself*, not in a trained class index.

### Why This Enables OW

The model **does not need to have "seen" a category during training to predict it at eval time.** It works through sketch similarity:
- **Training:** learns to match image regions against *seen-category sketches* (COCO train split).
- **Eval:** applies the same matching logic to *unseen-category sketches* (COCO val split, held-out cats). The sketch provides the category identity; the model just learns general region-to-sketch alignment.

This is fundamentally different from closed-set detectors (e.g., a linear head over 80 COCO classes), which have zero outputs for unseen categories.

### What's Missing: Data Partitioning

The **only missing piece** is the dataset-level filtering that enforces the train/eval split:
- **Train split**: should only include annotations for the 42 *seen* categories.
- **Val split**: should only include annotations for the 14 *unseen* categories.

Currently, LocFormer's `datasets/coco.py` doesn't do this — it loads all 56 COCO categories for both train and val. The i%4==0 unseen-cat split is mentioned in a comment (line 160-161) but **not implemented**.

---

## The 14 Held-Out Categories (i%4==0)

From 56 COCO category ordering (same as clip_ddetr):
```
bicycle, train, stop sign, dog, elephant, backpack, skateboard, knife, 
sandwich, pizza, couch, mouse, oven, clock
```

These are fixed by the `ALL_CATEGORIES` list index ordering. **Must verify LocFormer's `ALL_CATEGORIES` list has the same 56-entry ordering as clip_ddetr** to ensure the i%4==0 filter selects the same 14 cats.

---

## Required Code Changes

### File: `datasets/coco.py`

#### 1. **Partition `ALL_CATEGORIES` (add after line 163)**

```python
# Before (current, line 163-173):
ALL_CATEGORIES = [
    "aeroplane", "bicycle", "bird", "boat", ..., "zebra"  # 56 total
]

# After: keep ALL_CATEGORIES unchanged, add:
UNSEEN_CATEGORY_INDICES = [i for i in range(len(ALL_CATEGORIES)) if i % 4 == 0]
UNSEEN_CATEGORIES = {ALL_CATEGORIES[i] for i in UNSEEN_CATEGORY_INDICES}
SEEN_CATEGORIES = {ALL_CATEGORIES[i] for i in range(len(ALL_CATEGORIES)) if i % 4 != 0}
```

#### 2. **Update `__getitem__` to filter annotations by split (modify line 190-200)**

Currently, `build(image_set="train")` loads all 56 categories. Change to filter:

```python
@property
def _load_annotations(self):
    """Load and filter COCO annotations based on image_set (train/val) and category split."""
    coco = COCO(self.annotation_file)
    
    # Determine which categories are visible in this split
    if self.image_set == "train":
        visible_cats = SEEN_CATEGORIES  # Only seen cats in training
    else:  # val / test
        visible_cats = UNSEEN_CATEGORIES  # Only unseen cats in evaluation
    
    # Filter: keep only images/annotations with visible categories
    visible_cat_ids = {cat_id for cat_id, cat_name in coco.cats.items() 
                       if coco.cats[cat_id]['name'] in visible_cats}
    
    filtered_img_ids = set()
    for ann in coco.dataset['annotations']:
        if ann['category_id'] in visible_cat_ids:
            filtered_img_ids.add(ann['image_id'])
    
    # Rebuild the coco object with only visible images/annotations
    coco.imgs = {k: v for k, v in coco.imgs.items() if k in filtered_img_ids}
    coco.dataset['images'] = [img for img in coco.dataset['images'] 
                              if img['id'] in filtered_img_ids]
    coco.dataset['annotations'] = [ann for ann in coco.dataset['annotations'] 
                                    if ann['category_id'] in visible_cat_ids]
    coco.createIndex()  # Rebuild internal indices
    
    return coco
```

#### 3. **Expose one-shot mode via args (modify `build()` function, line ~501)**

Currently `num_sketches=5` is hardcoded. Add an arg:

```python
def build(image_set, args):
    # ... existing code ...
    num_sketches = getattr(args, 'num_sketches', 5)  # Default 5, override via --num_sketches 1
    dataset = CocoDetectionQD(
        img_folder, ann_file, 
        num_sketches=num_sketches,
        # ... rest of args ...
    )
    return dataset
```

#### 4. **Verify `qd_root` points to QuickDraw valid splits (line ~175)**

Confirm the path is set to the `.../sketchrnn/` directory with valid-split sketches:
```python
qd_root = os.path.join(data_root, "quickdraw", "sketchrnn")  # Should have .valid/ subdir
```

---

## Testing Checklist

After implementing the changes:

1. **Verify partition logic:**
   - Run with `image_set="train"` → check that annotations only cover SEEN categories (42 cats).
   - Run with `image_set="val"` → check that annotations only cover UNSEEN categories (14 cats).
   - Print sample category counts per split to confirm filter is working.

2. **Sketch availability:**
   - Confirm QuickDraw source (`qd_root/.../sketchrnn/`) has sketches for all 14 unseen cats (should auto-fallback with `has_class()` if not, but verify).

3. **One-shot mode:**
   - Add `--num_sketches 1` to training args, confirm training uses single sketch per category (reproducible via `random.seed(14)` in val).

4. **Per-epoch eval:**
   - Run a short training (2–3 epochs) with the OW split.
   - Confirm eval runs every epoch on `image_set="val"` (unseen cats only).
   - Check that mAP is meaningful (should be lower than closed-world, since unseen cats are harder).

5. **Reproduce paper numbers:**
   - Full 50-epoch training with OW split should converge to **~12.2 mAP** on unseen cats (Table 2, "Ours").
   - Compare against closed-world baseline (vanilla_rn50 run, 50 epochs, all 56 cats) to confirm the OW setting is working.

---

## Integration Notes

### Command-Line Args (update `main.py` or config)

Add these to argparse:
```python
parser.add_argument('--num_sketches', type=int, default=5, 
                    help='Number of sketches per category (1 for one-shot OW eval)')
parser.add_argument('--train_scheme_world', type=str, choices=['open', 'closed'], default='closed',
                    help='Use open-world train/val split (seen/unseen categories)')
```

Then pass to `build()`:
```python
if args.train_scheme_world == 'open':
    train_set = build('train', args)
    val_set = build('val', args)
else:  # closed
    train_set = build('train', args)
    val_set = build('val', args)  # Currently loads all categories; no change needed for closed-world
```

### Training Script Example

```bash
python main.py \
    --method vidt \
    --backbone_name swin_tiny \
    --epochs 50 \
    --num_sketches 1 \
    --train_scheme_world open \
    --coco_path /path/to/coco \
    --output_dir /path/to/outputs/lf_ow_rn50 \
    # ... rest of args ...
```

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Category mismatch (i%4==0 selects different 14 cats in LocFormer vs clip_ddetr) | Low | Verify `ALL_CATEGORIES` ordering matches before running. Hardcode the 14 cat names if ordering differs. |
| Missing sketches in QuickDraw for unseen cats | Very Low | QuickDraw has 345+ classes, covers all COCO categories. The `has_class()` fallback (line 250-254) should handle edge cases. |
| Eval mAP drops unexpectedly low | Medium | Expected — unseen cats are harder. But if mAP is ~0 or random noise, check that sketches are being loaded and model is forward-passing correctly. |
| Per-epoch eval time increases | Low | Eval is already per-epoch. No additional overhead. |
| Closed-world baseline breaks | Low | Changes are isolated to annotation filtering. Closed-world code paths should be unaffected. Test both splits. |

---

## Success Criteria

✅ **Implementation complete:**
- [ ] Partition logic added to `datasets/coco.py`.
- [ ] Args added to main.py (--num_sketches, --train_scheme_world).
- [ ] Test run with 2–3 epochs, --train_scheme_world open.
- [ ] Per-epoch eval produces non-zero mAP on unseen cats.
- [ ] Full 50-epoch OW training reaches ~12.2 mAP (matches paper Table 2, "Ours").

---

## Next Steps

1. **Implement** the `datasets/coco.py` changes (partition logic, annotation filtering).
2. **Test** with a short 2–3 epoch run on OW split.
3. **Verify** eval metrics and sketch loading.
4. **Launch** full 50-epoch OW training (expect ~4–5 hours on H200 / 2×ADA).
5. **Compare** against closed-world baseline (vanilla_rn50, already complete) and clip_ddetr OW results.
6. **Document** results and update LocFormer README with OW training instructions.

---

## Reference Materials

- **LocFormer paper:** Table 2, "Results in open-world, one-shot setting"
- **clip_ddetr OW implementation:** `/home/anirban/arkahaldi/clip_ddetr/datasets/coco.py` (reference for filtering logic)
- **Current LocFormer runs:** `/mnt/1tb/arka/locformer-SGOL/outputs/` (vanilla_rn50 closed-world baseline ready)
- **QuickDraw data:** `data/quickdraw/sketchrnn/` (345+ categories, .valid/ splits available)

---

## Contact / Questions

- Arch decision: binary head + sketch conditioning already supports OW (no model changes).
- Implementation: Data-side filtering in `datasets/coco.py` (~30 lines).
- Expected outcome: Reproduce paper's OW numbers (~12.2 mAP on unseen COCO cats).
