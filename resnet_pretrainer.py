import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import pickle as pkl
from typing import Optional
import lmdb, struct, io
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.models import resnet50, ResNet50_Weights
import wandb

from datasets.coco import convert_to_np_raw
import util.misc as utils
from dotenv import load_dotenv
load_dotenv()

COCO_HOME = Path(os.environ.get("COCO_HOME"))
PROJECT_HOME = Path(os.environ.get("PROJECT_HOME"))
QUICKDRAW_PATH = PROJECT_HOME / "checkpoints/processed_quick_draw_paths_purified.pkl"
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
WANDB_MODE = os.environ.get("WANDB_MODE")
WANDB_DISABLE_SERVICE = os.environ.get("WANDB_DISABLE_SERVICE")

UNSEEN_CATS = {
    "elephant",
    "bus",
    "bed",
    "sandwich",
    "umbrella",
    "toothbrush",
    "toaster",
    "donut",
    "oven",
    "fire hydrant",
    "apple",
    "car",
    "backpack",
    "skateboard",
}



@dataclass
class JobConfig:
    job_id: int
    seed: int
    image_set: str
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    weight_decay: float
    log_every: int
    momentum: float
    project: str
    entity: Optional[str]
    run_name: str
    device: str
    wandb_mode: str
    checkpoint_dir: str
    resume_from: Optional[str]


CHECKPOINT_FILENAME = "latest.pth"
META_LEN_KEY = b"__len__"

def _get_model_state(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def _load_model_state(model, state_dict):
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state_dict)


def save_checkpoint(model, optimizer, scaler, epoch, global_step, checkpoint_dir):
    checkpoint_path = Path(checkpoint_dir) / CHECKPOINT_FILENAME
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": _get_model_state(model),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else {},
    }
    torch.save(state, checkpoint_path)


def _resolve_resume_path(checkpoint_dir, resume_from):
    if resume_from is None:
        return None
    if resume_from == "latest":
        return Path(checkpoint_dir) / CHECKPOINT_FILENAME
    return Path(resume_from)


def load_checkpoint_if_available(model, optimizer, scaler, checkpoint_dir, resume_from):
    resume_path = _resolve_resume_path(checkpoint_dir, resume_from)
    if resume_path is None:
        return 0, 0
    should_notify = (
        not utils.is_dist_avail_and_initialized()
        or utils.is_main_process()
    )
    if not resume_path.exists():
        if should_notify:
            print(
                f"[{checkpoint_dir}] no checkpoint found at {resume_path}, starting fresh.",
                flush=True,
            )
        return 0, 0
    checkpoint = torch.load(resume_path, map_location="cpu")
    _load_model_state(model, checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scaler is not None and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
    next_epoch = checkpoint.get("epoch", 0) + 1
    global_step = checkpoint.get("global_step", 0)
    if should_notify:
        print(
            f"[{checkpoint_dir}] resumed {resume_path.name} at epoch {next_epoch} "
            f"(global step {global_step})",
            flush=True,
        )
    return next_epoch, global_step

def decode_record(buf: bytes):
    (lbl_len,) = struct.unpack("<H", buf[:2])
    lbl = buf[2:2+lbl_len]
    png = buf[2+lbl_len:]
    return lbl, png

class QuickDrawLMDB(Dataset):
    def __init__(self, lmdb_path: str, transform=None, class_to_idx=None, readonly=True):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.readonly = readonly
        self._env = None
        self._length = None

    def _open(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                subdir=False,
                readonly=self.readonly,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=2048,
            )
        if self._length is None:
            with self._env.begin(write=False) as txn:
                n = txn.get(META_LEN_KEY)
                if n is None:
                    raise RuntimeError("LMDB missing __len__")
                self._length = int(n.decode("utf-8"))
        return self._env

    def __len__(self):
        self._open()
        return self._length

    def __getitem__(self, idx):
        env = self._open()
        key = f"{idx:09d}".encode("utf-8")  # matches your tar keys
        with env.begin(write=False) as txn:
            buf = txn.get(key)
        if buf is None:
            raise IndexError(idx)

        lbl_b, png_b = decode_record(buf)
        lbl_s = lbl_b.decode("utf-8").strip()

        img = Image.open(io.BytesIO(png_b)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.class_to_idx is not None:
            y = self.class_to_idx[lbl_s]
        else:
            y = lbl_s

        return img, y

class QuickDrawSketchDataset(Dataset):
    def __init__(self, sketch_paths, class_to_idx, transform, coco_home):
        self.sketch_paths = sketch_paths
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.coco_home = coco_home

    def __len__(self):
        return len(self.sketch_paths)

    def _resolve_path(self, sketch_path):
        sketch_path = Path(sketch_path)
        if sketch_path.is_absolute():
            return Path(*self.coco_home.parts, *sketch_path.parts[4:])
        return sketch_path

    def __getitem__(self, idx):
        sketch_path = self._resolve_path(self.sketch_paths[idx])
        with open(sketch_path, "rb") as handle:
            sketch = pkl.load(handle)
        key = list(sketch.keys())[0]
        sketch = convert_to_np_raw(sketch[key])
        sketch = Image.fromarray(255 - np.asarray(sketch))
        if self.transform is not None:
            sketch = self.transform(sketch)
        label = self.class_to_idx[sketch_path.parent.name]
        return sketch, label



def load_quickdraw_paths(image_set):
    with open(QUICKDRAW_PATH, "rb") as handle:
        quickdraw_paths = pkl.load(handle)
    selected_paths = quickdraw_paths["train_x"] if image_set == "train" else quickdraw_paths["valid_x"]
    seen_paths = []
    for path in selected_paths:
        cat = Path(path).parent.name
        if cat not in UNSEEN_CATS:
            seen_paths.append(path)
    return seen_paths


def build_class_mapping(sketch_paths):
    classes = sorted({Path(path).parent.name for path in sketch_paths})
    return {cat: idx for idx, cat in enumerate(classes)}

def build_dataset(sketch_paths, class_to_idx):
    transforms_sketch = ResNet50_Weights.IMAGENET1K_V1.transforms()
    return QuickDrawLMDB(
        sketch_paths=sketch_paths,
        class_to_idx=class_to_idx,
        transform=transforms_sketch,
        coco_home=COCO_HOME,
    )

def build_dataloader(
    sketch_paths,
    class_to_idx,
    batch_size,
    num_workers,
    shuffle,
    drop_last,
    sampler=None,
    dataset=None,
):
    if dataset is None:
        dataset = build_dataset(sketch_paths, class_to_idx)
    if sampler is not None:
        shuffle = False
    loader_kwargs = dict(
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
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)
    return loader



def build_model(num_classes, device):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device, memory_format=torch.channels_last)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    log_every,
    run,
    global_step,
    scaler,
    use_amp,
    run_name,
    return_sums=False,
):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_correct = 0
    total = 0
    total_steps = len(loader)

    for step, (images, labels) in enumerate(loader):
        images = images.to(
            device, non_blocking=True, memory_format=torch.channels_last
        )
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss_value = loss.item()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss_value * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += images.size(0)

        global_step += 1
        if global_step % log_every == 0:
            batch_acc = (preds == labels).float().mean().item()
            log_data = {
                "train/loss": loss_value,
                "train/acc": batch_acc,
                "train/epoch": epoch,
                "train/step": global_step,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            if run is not None:
                run.log(log_data)
            print(
                f"[{run_name or 'local'}] step {global_step} ({step+1}/{total_steps} this epoch) | "
                f"loss {loss_value:.4f} acc {batch_acc:.4f} | "
                f"epoch {epoch+1} | lr {optimizer.param_groups[0]['lr']:.3e}",
                flush=True,
            )

    if return_sums:
        return running_loss, running_correct, total, global_step
    epoch_loss = running_loss / max(total, 1)
    epoch_acc = running_correct / max(total, 1)
    return epoch_loss, epoch_acc, global_step


@torch.no_grad()
def evaluate_one_epoch(model, loader, device, epoch, run, return_sums=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(
            device, non_blocking=True, memory_format=torch.channels_last
        )
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += images.size(0)

    if return_sums:
        return running_loss, running_correct, total
    epoch_loss = running_loss / max(total, 1)
    epoch_acc = running_correct / max(total, 1)
    if run is not None:
        run.log(
            {
                "val/epoch_loss": epoch_loss,
                "val/epoch_acc": epoch_acc,
                "val/epoch": epoch,
            }
        )
    return epoch_loss, epoch_acc


def reduce_epoch_stats(loss_sum, correct_sum, total_sum, device):
    if not utils.is_dist_avail_and_initialized():
        return loss_sum, correct_sum, total_sum
    tensor = torch.tensor(
        [loss_sum, correct_sum, total_sum], device=device, dtype=torch.float64
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor[0].item(), tensor[1].item(), tensor[2].item()



def train_job(job_cfg):
    if job_cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(job_cfg.device))
    torch.backends.cudnn.benchmark = True
    device = torch.device(job_cfg.device)

    set_seed(job_cfg.seed)
    run = None
    if job_cfg.job_id == 0 and WANDB_MODE != "disabled":
        run = wandb.init(
            project=job_cfg.project,
            entity=job_cfg.entity,
            name=job_cfg.run_name,
            config=vars(job_cfg),
            reinit=True,
        )

    train_paths = load_quickdraw_paths(job_cfg.image_set)
    val_paths = load_quickdraw_paths("valid")
    train_cls_to_idx = build_class_mapping(train_paths)
    val_cls_to_idx = build_class_mapping(val_paths)
    if set(train_cls_to_idx.keys()) != set(val_cls_to_idx.keys()):
        raise ValueError("Train/val class sets do not match.")
    val_cls_to_idx = {cat: train_cls_to_idx[cat] for cat in train_cls_to_idx}

    loader = build_dataloader(
        sketch_paths=train_paths,
        class_to_idx=train_cls_to_idx,
        batch_size=job_cfg.batch_size,
        num_workers=job_cfg.num_workers,
        shuffle=True,
        drop_last=True,
    )
    val_loader = build_dataloader(
        sketch_paths=val_paths,
        class_to_idx=val_cls_to_idx,
        batch_size=job_cfg.batch_size,
        num_workers=job_cfg.num_workers,
        shuffle=False,
        drop_last=False,
    )
    model = build_model(len(train_cls_to_idx), device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=job_cfg.lr, weight_decay=job_cfg.weight_decay
    )
    use_amp = job_cfg.device.startswith("cuda") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint_root = Path(job_cfg.checkpoint_dir) / job_cfg.run_name
    start_epoch, global_step = load_checkpoint_if_available(
        model, optimizer, scaler, checkpoint_root, job_cfg.resume_from
    )
    print(
        f"[{job_cfg.run_name}] starting training from epoch {start_epoch}/{job_cfg.epochs} "
        f"checkpoint_dir={checkpoint_root}",
        flush=True,
    )

    epoch_loss = 0.0
    epoch_acc = 0.0
    for epoch in range(start_epoch, job_cfg.epochs):
        print(
            f"[{job_cfg.run_name}] epoch {epoch + 1}/{job_cfg.epochs} begin",
            flush=True,
        )
        epoch_loss, epoch_acc, global_step = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            epoch,
            job_cfg.log_every,
            run,
            global_step,
            scaler,
            use_amp,
            job_cfg.run_name,
        )
        val_loss, val_acc = evaluate_one_epoch(model, val_loader, device, epoch, run)
        print(
            f"[{job_cfg.run_name}] epoch {epoch + 1}/{job_cfg.epochs} done | "
            f"train loss {epoch_loss:.4f} acc {epoch_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | global step {global_step}",
            flush=True,
        )
        if run is not None:
            run.log(
                {
                    "train/epoch_loss": epoch_loss,
                    "train/epoch_acc": epoch_acc,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
            )
        save_checkpoint(model, optimizer, scaler, epoch, global_step, checkpoint_root)

    if run is not None:
        run.finish()

    return {
        "job_id": job_cfg.job_id,
        "final_loss": epoch_loss,
        "final_acc": epoch_acc,
        "global_step": global_step,
    }



def run_parallel_jobs(job_cfgs, max_workers):
    if len(job_cfgs) == 1:
        return [train_job(job_cfgs[0])]

    ctx = torch.multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(train_job, job_cfg) for job_cfg in job_cfgs]
        for future in as_completed(futures):
            results.append(future.result())
    return results



def build_job_configs(args):
    device_count = torch.cuda.device_count()
    job_cfgs = []
    for job_id in range(args.job_count):
        seed = args.seed + job_id
        if args.device.startswith("cuda"):
            if device_count > 0:
                device = f"cuda:{job_id % device_count}"
            else:
                device = "cpu"
        else:
            device = args.device
        run_name = f"{args.run_name}-job{job_id}" if args.job_count > 1 else args.run_name
        job_cfgs.append(
            JobConfig(
                job_id=job_id,
                seed=seed,
                image_set=args.image_set,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                epochs=args.epochs,
                momentum=args.momentum,
                lr=args.lr,
                weight_decay=args.weight_decay,
                log_every=args.log_every,
                project=args.wandb_project,
                entity=args.wandb_entity,
                run_name=run_name,
                device=device,
                wandb_mode=WANDB_MODE,
                checkpoint_dir=args.checkpoint_dir,
                resume_from=args.resume_from,
            )
        )
    return job_cfgs


def train_ddp(args):
    utils.init_distributed_mode(args)
    torch.backends.cudnn.benchmark = True
    if args.distributed and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device(args.device)

    set_seed(args.seed + utils.get_rank())
    run = None
    if utils.is_main_process() and WANDB_MODE != "disabled":
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
            reinit=True,
        )
    args.lr = args.lr * (args.batch_size * utils.get_world_size() / 256)
    train_paths = load_quickdraw_paths("train")
    val_paths = load_quickdraw_paths("valid")
    train_cls_to_idx = build_class_mapping(train_paths)
    val_cls_to_idx = build_class_mapping(val_paths)
    if set(train_cls_to_idx.keys()) != set(val_cls_to_idx.keys()):
        raise ValueError("Train/val class sets do not match.")
    val_cls_to_idx = {cat: train_cls_to_idx[cat] for cat in train_cls_to_idx}

    train_dataset = build_dataset(train_paths, train_cls_to_idx)
    val_dataset = build_dataset(val_paths, val_cls_to_idx)

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True) if args.distributed else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
    )

    train_loader = build_dataloader(
        sketch_paths=train_paths,
        class_to_idx=train_cls_to_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        sampler=train_sampler,
        dataset=train_dataset,
    )
    val_loader = build_dataloader(
        sketch_paths=val_paths,
        class_to_idx=val_cls_to_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        sampler=val_sampler,
        dataset=val_dataset,
    )
    if utils.is_main_process():
        print(
            f"Dataset size: {len(train_dataset)} train samples, {len(val_dataset)} val samples.",
            flush=True,
        )
    model = build_model(len(train_cls_to_idx), device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint_root = Path(args.checkpoint_dir) / args.run_name
    start_epoch, global_step = load_checkpoint_if_available(
        model, optimizer, scaler, checkpoint_root, args.resume_from
    )
    if utils.is_main_process():
        print(
            f"[{args.run_name}] starting DDP training from epoch {start_epoch}/{args.epochs} "
            f"checkpoint_dir={checkpoint_root}",
            flush=True,
        )
    try:
        for epoch in range(start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            loss_sum, correct_sum, total_sum, global_step = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch,
                args.log_every,
                run,
                global_step,
                scaler,
                use_amp,
                args.run_name,
                return_sums=True,
            )
            loss_sum, correct_sum, total_sum = reduce_epoch_stats(
                loss_sum, correct_sum, total_sum, device
            )
            epoch_loss = loss_sum / max(total_sum, 1)
            epoch_acc = correct_sum / max(total_sum, 1)

            if run is not None:
                run.log(
                    {
                        "train/epoch_loss": epoch_loss,
                        "train/epoch_acc": epoch_acc,
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    }
                )

            val_loss_sum, val_correct_sum, val_total_sum = evaluate_one_epoch(
                model,
                val_loader,
                device,
                epoch,
                run,
                return_sums=True,
            )
            val_loss_sum, val_correct_sum, val_total_sum = reduce_epoch_stats(
                val_loss_sum, val_correct_sum, val_total_sum, device
            )
            val_epoch_loss = val_loss_sum / max(val_total_sum, 1)
            val_epoch_acc = val_correct_sum / max(val_total_sum, 1)

            if run is not None:
                run.log(
                    {
                        "val/epoch_loss": val_epoch_loss,
                        "val/epoch_acc": val_epoch_acc,
                        "val/epoch": epoch,
                    }
                )

        if utils.is_main_process():
            print(
                f"[{args.run_name}] epoch {epoch + 1}/{args.epochs} | "
                f"train loss {epoch_loss:.4f} acc {epoch_acc:.4f} | "
                f"val loss {val_epoch_loss:.4f} acc {val_epoch_acc:.4f} | "
                f"global step {global_step}",
                flush=True,
            )
            save_checkpoint(model, optimizer, scaler, epoch, global_step, checkpoint_root)

        if utils.is_dist_avail_and_initialized():
            dist.barrier()
    finally:
        if run is not None:
            run.finish()
        if utils.is_dist_avail_and_initialized():
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet QuickDraw pretrainer")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.1) # linear scaling rule => lr * (n * batch_size / 256)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--job-count", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default="r50-adamw")
    parser.add_argument("--wandb-project", default="resnet50-quickdraw-pt")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint path or 'latest' to resume training from.",
    )
    parser.add_argument("--ddp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ddp:
        train_ddp(args)
    else:
        job_cfgs = build_job_configs(args)
        max_workers = min(args.max_workers, len(job_cfgs))
        run_parallel_jobs(job_cfgs, max_workers=max_workers)


if __name__ == "__main__":
    main()

# """

# """
