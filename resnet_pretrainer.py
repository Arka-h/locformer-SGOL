import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.models import resnet50, ResNet50_Weights
import wandb

from datasets.coco import rasterize_stroke3
import util.misc as utils
from dotenv import load_dotenv

load_dotenv()

QD_ROOT      = Path(os.environ["QD_DATASET"])
PROJECT_HOME = Path(os.environ["PROJECT_HOME"])
WANDB_MODE   = os.environ.get("WANDB_MODE", "disabled")

UNSEEN_CATS = {
    "elephant", "bus", "bed", "sandwich", "umbrella", "toothbrush",
    "toaster", "donut", "oven", "fire hydrant", "apple", "car",
    "backpack", "skateboard",
}

CHECKPOINT_FILENAME      = "latest.pth"
BEST_CHECKPOINT_FILENAME = "best.pth"


# ---------------------------------------------------------------------------
# QuickDraw mmap dataset — mirrors SLIP's QuickDraw + QD_CLIP
# ---------------------------------------------------------------------------
def get_class_names(root: Path, split: str, exclude: set = frozenset()) -> list:
    """
    Discover classes by scanning for {class}.{split}.ptr.npy files under root.
    Returns a sorted list, excluding any names in `exclude`.
    """
    names = []
    for f in sorted(root.iterdir()):
        if f.name.endswith(f'.{split}.ptr.npy'):
            name = f.name[: -(len(f'.{split}.ptr.npy'))]
            if name not in exclude:
                names.append(name)
    return names


class QuickDrawSketchDataset(Dataset):
    """
    Mmap-backed QuickDraw classification dataset.

    Mirrors SLIP's QuickDraw base class.  Classes are loaded in parallel via
    ThreadPoolExecutor (avoids long NFS startup delays).  self.samples is a
    numpy (N, 2) int32 array of [class_idx, sample_idx] rows — shared
    read-only across DataLoader workers after fork, no CoW copies.

    Files expected under `root`:
        {class_name}.{split}.ptr.npy      — shape (N+1,) int64
        {class_name}.{split}.strokes.npy  — shape (total_strokes, 3) int16
    """

    def __init__(self, root: Path, class_names: list, split: str,
                 class_to_idx: dict, transform=None):
        assert split in ('train', 'valid', 'test'), \
            f"split must be 'train', 'valid', or 'test', got {split!r}"
        self.root         = root
        self.split        = split
        self.class_names  = list(class_names)
        self.class_to_idx = class_to_idx
        # NOTE: ImageNet stats — sketch-specific normalization would be marginally better
        self.transform    = transform
        # class_name -> (ptr ndarray, strokes mmap)
        self._mmap_cache: dict = {}
        # idx -> class_name inverse of class_to_idx, populated only for loaded classes
        self.idx_to_class: dict = {}

        def _load_one(args):
            class_name, class_idx = args
            ptr_path     = root / f'{class_name}.{split}.ptr.npy'
            strokes_path = root / f'{class_name}.{split}.strokes.npy'
            if not (ptr_path.exists() and strokes_path.exists()):
                return class_name, class_idx, None, None
            ptr     = np.load(ptr_path)
            strokes = np.load(strokes_path, mmap_mode='r')
            return class_name, class_idx, ptr, strokes

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=32) as ex:
            results = list(ex.map(_load_one,
                                  [(n, class_to_idx[n]) for n in class_names]))

        missing, class_cols, sample_cols = [], [], []
        for class_name, class_idx, ptr, strokes in results:
            if ptr is None:
                missing.append(class_name)
                continue
            self._mmap_cache[class_name]  = (ptr, strokes)
            self.idx_to_class[class_idx]  = class_name
            n = len(ptr) - 1
            class_cols.append(np.full(n, class_idx, dtype=np.int32))
            sample_cols.append(np.arange(n,          dtype=np.int32))

        # numpy array: shared read-only across workers, no CoW on fork
        self.samples = np.stack(
            [np.concatenate(class_cols), np.concatenate(sample_cols)], axis=1
        )

        if missing:
            print(f'[QuickDrawSketchDataset] Warning: missing npy files for '
                  f'{len(missing)} class(es): {missing[:5]}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        class_idx, sample_idx = self.samples[idx]
        class_name = self.idx_to_class[int(class_idx)]
        ptr, strokes = self._mmap_cache[class_name]
        stroke_seq   = strokes[ptr[sample_idx]: ptr[sample_idx + 1]]  # (T, 3) int16
        img = rasterize_stroke3(stroke_seq)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(class_idx)


# ---------------------------------------------------------------------------
# Dataset / DataLoader builders
# ---------------------------------------------------------------------------
def build_dataset(root: Path, split: str, class_names: list,
                  class_to_idx: dict) -> QuickDrawSketchDataset:
    transform = ResNet50_Weights.IMAGENET1K_V1.transforms()
    return QuickDrawSketchDataset(
        root=root,
        class_names=class_names,
        split=split,
        class_to_idx=class_to_idx,
        transform=transform,
    )


def build_dataloader(dataset, batch_size, num_workers, shuffle,
                     drop_last, sampler=None):
    if sampler is not None:
        shuffle = False
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(num_classes, device):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device, memory_format=torch.channels_last)


# ---------------------------------------------------------------------------
# LR helpers
# ---------------------------------------------------------------------------
def _scale_lr(base_lr: float, batch_size: int, world_size: int) -> float:
    """
    Linear LR scaling rule (Goyal et al. 2017).
    `batch_size` is per-process (per-GPU), not the global batch.
    Effective global batch = batch_size × world_size.
    Pass world_size=1 for single-process / parallel-jobs runs.
    """
    return base_lr * (batch_size * world_size / 256)


def build_scheduler(optimizer, epochs: int, warmup_epochs: int):
    """
    Cosine annealing with optional linear warmup.
    For large scaled LRs (> ~0.2), warmup_epochs ≥ 5 is recommended to
    avoid loss spikes at the start of training.
    """
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1),
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _get_model_state(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def _load_model_state(model, state_dict):
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state_dict)


def save_checkpoint(model, optimizer, scheduler, scaler,
                    epoch, global_step, best_val_acc, checkpoint_dir):
    """Full training state — used for resume."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":           epoch,
        "global_step":     global_step,
        "best_val_acc":    best_val_acc,
        "model_state":     _get_model_state(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state":    scaler.state_dict() if scaler is not None else {},
    }, Path(checkpoint_dir) / CHECKPOINT_FILENAME)


def save_best_checkpoint(model, val_acc, epoch, checkpoint_dir):
    """Model state only — for loading into Detector.sketch_embedding downstream."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": _get_model_state(model),
        "val_acc":     val_acc,
        "epoch":       epoch,
    }, Path(checkpoint_dir) / BEST_CHECKPOINT_FILENAME)


def load_checkpoint_if_available(model, optimizer, scheduler, scaler,
                                  checkpoint_dir, resume_from):
    """Returns (start_epoch, global_step, best_val_acc)."""
    if resume_from is None:
        return 0, 0, 0.0
    path = (Path(checkpoint_dir) / CHECKPOINT_FILENAME
            if resume_from == "latest" else Path(resume_from))
    notify = not utils.is_dist_avail_and_initialized() or utils.is_main_process()
    if not path.exists():
        if notify:
            print(f"[checkpoint] no file at {path}, starting fresh.", flush=True)
        return 0, 0, 0.0
    ckpt = torch.load(path, map_location="cpu")
    _load_model_state(model, ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    start_epoch  = ckpt.get("epoch", 0) + 1
    global_step  = ckpt.get("global_step", 0)
    best_val_acc = ckpt.get("best_val_acc", 0.0)
    if notify:
        print(f"[checkpoint] resumed {path.name} at epoch {start_epoch} "
              f"(step {global_step}, best_val_acc {best_val_acc:.4f})", flush=True)
    return start_epoch, global_step, best_val_acc


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_class_to_idx(train_class_names, val_class_names):
    # sorted() guarantees deterministic mapping — do not change ordering
    # or checkpoints will be invalidated (fc layer shape depends on num_classes)
    return {name: i for i, name in enumerate(
        sorted(set(train_class_names) | set(val_class_names)))}


# ---------------------------------------------------------------------------
# Training / eval loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, epoch, log_every,
                    run, global_step, scaler, use_amp, run_name,
                    return_sums=False):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss, running_correct, total = 0.0, 0, 0
    total_steps = len(loader)

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        loss_value = loss.item()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss    += loss_value * images.size(0)
        preds            = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total           += images.size(0)
        global_step     += 1

        if global_step % log_every == 0:
            batch_acc = (preds == labels).float().mean().item()
            log_data  = {
                "train/loss":  loss_value,
                "train/acc":   batch_acc,
                "train/epoch": epoch,
                "train/step":  global_step,
                "train/lr":    optimizer.param_groups[0]["lr"],
            }
            if run is not None:
                run.log(log_data)
            print(
                f"[{run_name or 'local'}] step {global_step} "
                f"({step+1}/{total_steps}) | "
                f"loss {loss_value:.4f} acc {batch_acc:.4f} | "
                f"epoch {epoch+1} | lr {optimizer.param_groups[0]['lr']:.3e}",
                flush=True,
            )

    if return_sums:
        return running_loss, running_correct, total, global_step
    return running_loss / max(total, 1), running_correct / max(total, 1), global_step


@torch.no_grad()
def evaluate_one_epoch(model, loader, device, epoch, run, return_sums=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss, running_correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        running_loss    += loss.item() * images.size(0)
        preds            = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total           += images.size(0)

    if return_sums:
        return running_loss, running_correct, total
    epoch_loss = running_loss / max(total, 1)
    epoch_acc  = running_correct / max(total, 1)
    if run is not None:
        run.log({"val/epoch_loss": epoch_loss, "val/epoch_acc": epoch_acc,
                 "val/epoch": epoch})
    return epoch_loss, epoch_acc


def reduce_epoch_stats(loss_sum, correct_sum, total_sum, device):
    if not utils.is_dist_avail_and_initialized():
        return loss_sum, correct_sum, total_sum
    tensor = torch.tensor(
        [loss_sum, correct_sum, total_sum], device=device, dtype=torch.float64
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor[0].item(), tensor[1].item(), tensor[2].item()


# ---------------------------------------------------------------------------
# Single-node parallel training
# ---------------------------------------------------------------------------
@dataclass
class JobConfig:
    job_id:         int
    seed:           int
    batch_size:     int
    num_workers:    int
    epochs:         int
    lr:             float
    weight_decay:   float
    log_every:      int
    momentum:       float
    warmup_epochs:  int
    project:        str
    entity:         Optional[str]
    run_name:       str
    device:         str
    wandb_mode:     str
    checkpoint_dir: str
    resume_from:    Optional[str]


def train_job(job_cfg):
    if job_cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(job_cfg.device))
    torch.backends.cudnn.benchmark = True
    device = torch.device(job_cfg.device)

    set_seed(job_cfg.seed)
    run = None
    if job_cfg.job_id == 0 and WANDB_MODE != "disabled":
        run = wandb.init(
            project=job_cfg.project, entity=job_cfg.entity,
            name=job_cfg.run_name, config=vars(job_cfg), reinit=True,
        )

    train_class_names = get_class_names(QD_ROOT, 'train', exclude=UNSEEN_CATS)
    val_class_names   = get_class_names(QD_ROOT, 'valid', exclude=UNSEEN_CATS)
    class_to_idx      = _build_class_to_idx(train_class_names, val_class_names)

    train_dataset = build_dataset(QD_ROOT, 'train', train_class_names, class_to_idx)
    val_dataset   = build_dataset(QD_ROOT, 'valid', val_class_names,   class_to_idx)

    loader     = build_dataloader(train_dataset, job_cfg.batch_size, job_cfg.num_workers,
                                  shuffle=True,  drop_last=True)
    val_loader = build_dataloader(val_dataset,   job_cfg.batch_size, job_cfg.num_workers,
                                  shuffle=False, drop_last=False)

    model = build_model(len(class_to_idx), device)
    # world_size=1: each parallel job is independent, not gradient-summed DDP
    effective_lr = _scale_lr(job_cfg.lr, job_cfg.batch_size, world_size=1)
    optimizer  = torch.optim.SGD(model.parameters(), lr=effective_lr,
                                 momentum=job_cfg.momentum, weight_decay=job_cfg.weight_decay)
    scheduler  = build_scheduler(optimizer, job_cfg.epochs, job_cfg.warmup_epochs)
    use_amp    = job_cfg.device.startswith("cuda") and torch.cuda.is_available()
    scaler     = torch.amp.GradScaler("cuda", enabled=use_amp)
    ckpt_root  = Path(job_cfg.checkpoint_dir) / job_cfg.run_name
    start_epoch, global_step, best_val_acc = load_checkpoint_if_available(
        model, optimizer, scheduler, scaler, ckpt_root, job_cfg.resume_from)

    print(f"[{job_cfg.run_name}] {len(train_dataset)} train / {len(val_dataset)} val | "
          f"{len(class_to_idx)} classes | lr {effective_lr:.4f} | "
          f"starting epoch {start_epoch}/{job_cfg.epochs}", flush=True)

    epoch_loss = epoch_acc = 0.0
    for epoch in range(start_epoch, job_cfg.epochs):
        epoch_loss, epoch_acc, global_step = train_one_epoch(
            model, loader, optimizer, device, epoch, job_cfg.log_every,
            run, global_step, scaler, use_amp, job_cfg.run_name)
        val_loss, val_acc = evaluate_one_epoch(model, val_loader, device, epoch, run)
        scheduler.step()

        print(
            f"[{job_cfg.run_name}] epoch {epoch+1}/{job_cfg.epochs} | "
            f"train loss {epoch_loss:.4f} acc {epoch_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | step {global_step} | "
            f"lr {optimizer.param_groups[0]['lr']:.3e}",
            flush=True,
        )
        if run is not None:
            run.log({"train/epoch_loss": epoch_loss, "train/epoch_acc": epoch_acc,
                     "train/epoch": epoch, "train/global_step": global_step,
                     "train/lr": optimizer.param_groups[0]["lr"],
                     "val/epoch_loss": val_loss, "val/epoch_acc": val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_best_checkpoint(model, val_acc, epoch, ckpt_root)
        save_checkpoint(model, optimizer, scheduler, scaler,
                        epoch, global_step, best_val_acc, ckpt_root)

    if run is not None:
        run.finish()
    return {"job_id": job_cfg.job_id, "final_loss": epoch_loss,
            "final_acc": epoch_acc, "global_step": global_step}


def run_parallel_jobs(job_cfgs, max_workers):
    if len(job_cfgs) == 1:
        return [train_job(job_cfgs[0])]
    ctx     = torch.multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(train_job, cfg) for cfg in job_cfgs]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def build_job_configs(args):
    device_count = torch.cuda.device_count()
    job_cfgs     = []
    for job_id in range(args.job_count):
        device   = (f"cuda:{job_id % max(device_count, 1)}"
                    if args.device.startswith("cuda") and device_count > 0
                    else "cpu")
        run_name = (f"{args.run_name}-job{job_id}"
                    if args.job_count > 1 else args.run_name)
        job_cfgs.append(JobConfig(
            job_id=job_id, seed=args.seed + job_id,
            batch_size=args.batch_size, num_workers=args.num_workers,
            epochs=args.epochs, momentum=args.momentum,
            lr=args.lr, weight_decay=args.weight_decay,
            log_every=args.log_every, warmup_epochs=args.warmup_epochs,
            project=args.wandb_project, entity=args.wandb_entity,
            run_name=run_name, device=device, wandb_mode=WANDB_MODE,
            checkpoint_dir=args.checkpoint_dir, resume_from=args.resume_from,
        ))
    return job_cfgs


# ---------------------------------------------------------------------------
# DDP training
# ---------------------------------------------------------------------------
def train_ddp(args):
    # args.gpu / args.distributed / args.rank / args.world_size are
    # injected by utils.init_distributed_mode(args) below
    utils.init_distributed_mode(args)
    torch.backends.cudnn.benchmark = True
    device = (torch.device(f"cuda:{args.gpu}")
              if args.distributed and torch.cuda.is_available()
              else torch.device(args.device))

    set_seed(args.seed + utils.get_rank())
    run = None
    if utils.is_main_process() and WANDB_MODE != "disabled":
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.run_name, config=vars(args), reinit=True,
        )

    # world_size = number of DDP processes; batch_size is per-GPU
    effective_lr = _scale_lr(args.lr, args.batch_size, utils.get_world_size())

    train_class_names = get_class_names(QD_ROOT, 'train', exclude=UNSEEN_CATS)
    val_class_names   = get_class_names(QD_ROOT, 'valid', exclude=UNSEEN_CATS)
    class_to_idx      = _build_class_to_idx(train_class_names, val_class_names)

    train_dataset = build_dataset(QD_ROOT, 'train', train_class_names, class_to_idx)
    val_dataset   = build_dataset(QD_ROOT, 'valid', val_class_names,   class_to_idx)

    if utils.is_main_process():
        print(f"Dataset: {len(train_dataset)} train / {len(val_dataset)} val | "
              f"{len(class_to_idx)} classes | effective_lr {effective_lr:.4f}", flush=True)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)  if args.distributed else None
    val_sampler   = DistributedSampler(val_dataset,   shuffle=False) if args.distributed else None

    train_loader = build_dataloader(train_dataset, args.batch_size, args.num_workers,
                                    shuffle=True,  drop_last=True,  sampler=train_sampler)
    val_loader   = build_dataloader(val_dataset,   args.batch_size, args.num_workers,
                                    shuffle=False, drop_last=False, sampler=val_sampler)

    model = build_model(len(class_to_idx), device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=effective_lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    use_amp   = device.type == "cuda"
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)
    ckpt_root = Path(args.checkpoint_dir) / args.run_name
    start_epoch, global_step, best_val_acc = load_checkpoint_if_available(
        model, optimizer, scheduler, scaler, ckpt_root, args.resume_from)

    try:
        for epoch in range(start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            loss_sum, correct_sum, total_sum, global_step = train_one_epoch(
                model, train_loader, optimizer, device, epoch, args.log_every,
                run, global_step, scaler, use_amp, args.run_name, return_sums=True)
            loss_sum, correct_sum, total_sum = reduce_epoch_stats(
                loss_sum, correct_sum, total_sum, device)
            epoch_loss = loss_sum    / max(total_sum, 1)
            epoch_acc  = correct_sum / max(total_sum, 1)

            val_loss_sum, val_correct_sum, val_total_sum = evaluate_one_epoch(
                model, val_loader, device, epoch, run, return_sums=True)
            val_loss_sum, val_correct_sum, val_total_sum = reduce_epoch_stats(
                val_loss_sum, val_correct_sum, val_total_sum, device)
            val_epoch_loss = val_loss_sum    / max(val_total_sum, 1)
            val_epoch_acc  = val_correct_sum / max(val_total_sum, 1)

            scheduler.step()

            if utils.is_main_process():
                print(
                    f"[{args.run_name}] epoch {epoch+1}/{args.epochs} | "
                    f"train loss {epoch_loss:.4f} acc {epoch_acc:.4f} | "
                    f"val loss {val_epoch_loss:.4f} acc {val_epoch_acc:.4f} | "
                    f"step {global_step} | lr {optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
                if run is not None:
                    run.log({"train/epoch_loss": epoch_loss, "train/epoch_acc": epoch_acc,
                             "train/epoch": epoch, "train/global_step": global_step,
                             "train/lr": optimizer.param_groups[0]["lr"],
                             "val/epoch_loss": val_epoch_loss,
                             "val/epoch_acc": val_epoch_acc, "val/epoch": epoch})
                if val_epoch_acc > best_val_acc:
                    best_val_acc = val_epoch_acc
                    save_best_checkpoint(model, val_epoch_acc, epoch, ckpt_root)
                save_checkpoint(model, optimizer, scheduler, scaler,
                                epoch, global_step, best_val_acc, ckpt_root)

        if utils.is_dist_avail_and_initialized():
            dist.barrier()
    finally:
        if run is not None:
            run.finish()
        if utils.is_dist_avail_and_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Argument parsing + entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="ResNet-50 QuickDraw pretrainer")
    parser.add_argument("--batch-size",      type=int,   default=1024)
    parser.add_argument("--num-workers",     type=int,   default=8)
    parser.add_argument("--epochs",          type=int,   default=40)
    parser.add_argument("--lr",              type=float, default=0.1,
                        help="Base LR (per-process); scaled by linear rule at runtime.")
    parser.add_argument("--weight-decay",    type=float, default=1e-4)
    parser.add_argument("--momentum",        type=float, default=0.9)
    parser.add_argument("--warmup-epochs",   type=int,   default=0,
                        help="Linear warmup epochs before cosine decay. "
                             "Recommended ≥5 for large scaled LRs (>0.2).")
    parser.add_argument("--seed",            type=int,   default=14)
    parser.add_argument("--job-count",       type=int,   default=1)
    parser.add_argument("--max-workers",     type=int,   default=2)
    parser.add_argument("--log-every",       type=int,   default=100)
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--run-name",        default="r50-sgd")
    parser.add_argument("--wandb-project",   default="resnet50-quickdraw-pt")
    parser.add_argument("--wandb-entity",    default=None)
    parser.add_argument("--checkpoint-dir",  default="checkpoints")
    parser.add_argument("--resume-from",     type=str,   default=None,
                        help="Checkpoint path or 'latest'.")
    parser.add_argument("--ddp",             action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ddp:
        train_ddp(args)
    else:
        job_cfgs    = build_job_configs(args)
        max_workers = min(args.max_workers, len(job_cfgs))
        run_parallel_jobs(job_cfgs, max_workers=max_workers)


if __name__ == "__main__":
    main()
