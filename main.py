# ------------------------------------------------------------------------
# DETR
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Additionally modified by NAVER Corp. for ViDT
# ------------------------------------------------------------------------

import os
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import resource
import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, train_one_epoch_with_teacher
from methods import build_model
from util.scheduler import create_scheduler
from arguments import get_args_parser
import argparse
import wandb

def _wandb_safe_config(args):
    cfg = {}
    for k, v in vars(args).items():
        # make sure Paths / weird types don't break serialization
        if isinstance(v, Path):
            cfg[k] = str(v)
        else:
            cfg[k] = v
    return cfg

def setup_wandb(args, run_name=None):
    if not utils.is_main_process():
        return None

    project = os.getenv("WANDB_PROJECT", "locformer-SGOL")
    entity  = os.getenv("WANDB_ENTITY", "aurkohaldi")

    run_id_path = None
    run_id = None
    if getattr(args, "output_dir", None):
        run_id_path = Path(args.output_dir) / "wandb_run_id.txt"
        if run_id_path.exists():
            run_id = run_id_path.read_text().strip() or None

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        id=run_id,
        resume="allow",
        config=_wandb_safe_config(args),
        dir=getattr(args, "output_dir", None),
        settings=wandb.Settings(start_method="thread"),
    )

    if run_id_path is not None and (run_id is None):
        run_id_path.write_text(run.id)

    # nice metric grouping
    wandb.define_metric("epoch")
    wandb.define_metric("train_*", step_metric="epoch")
    wandb.define_metric("test_*",  step_metric="epoch")
    wandb.define_metric("lr/*",    step_metric="epoch")
    wandb.define_metric("time/*",  step_metric="epoch")
    # per-step training curves keyed on global step
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")

    # optional (can be heavy):
    # wandb.watch(model, log="gradients", log_freq=200)

    return run

def build_distil_model(args):
    """ build a teacher model """
    assert args.distil_model in ['vidt_nano', 'vidt_tiny', 'vidt_small', 'vidt_base']
    return build_model(args, is_teacher=True)

def main(args):
    """ main function to train a ViDT model """

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # Gradient accumulation setup
    if args.n_iter_to_acc > 1:
        if args.batch_size % args.n_iter_to_acc != 0:
            print("Not supported divisor for acc grade.")
            import sys
            sys.exit(1)
        print("Gradient Accumulation is applied.")
        print("The batch: ", args.batch_size, "->", int(args.batch_size / args.n_iter_to_acc),
              'but updated every ', args.n_iter_to_acc, 'steps.')
        args.batch_size = args.batch_size // args.n_iter_to_acc
    ##

    # distributed data parallel setup
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)
    wandb_run = None
    if utils.is_main_process():
        run_name = os.path.basename(args.output_dir) if args.output_dir else None
        wandb_run = setup_wandb(args, run_name=run_name)
        wandb.config.update({"git_sha": utils.get_sha()}, allow_val_change=True)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # import pdb;pdb.set_trace()
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    teacher_model = None
    if args.distil_model is not None:
        teacher_model = build_distil_model(args)
        teacher_model.to(device)
        if args.distil_model_path:
            checkpoint = torch.load(args.distil_model_path, map_location='cpu')
            teacher_state = checkpoint.get('model', checkpoint)
            teacher_model.load_state_dict(teacher_state, strict=False)


    # parallel model setup
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module

        if teacher_model is not None:
            teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model,
                                                                      device_ids=[args.gpu],
                                                                      find_unused_parameters=True)

    # print parameter info.
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if teacher_model is not None:
        n_parameters = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
        print('number of params for teacher:', n_parameters)

    # optimizer setup
    def match_name_keywords(name, name_keywords):
        return any(kw in name for kw in name_keywords)

    def build_optimizer(model, args):
        # Standard Deformable-DETR / ViDT param grouping (4 groups):
        #   head        : non-backbone, non-linear-proj (decoder heads, sketch encoder, fusion) -> args.lr (1e-4)
        #   backbone_*  : Swin backbone, fine-tuned at args.lr_backbone (1e-5, 10x lower); norms/biases get wd=0
        #   linear_proj : deformable sampling_offsets / reference_points -> args.lr * args.lr_linear_proj_mult (1e-5)
        # Previously the backbone ran at args.lr and there was NO linear_proj group (both args were dead).
        skip = model.backbone.no_weight_decay() if hasattr(model.backbone, 'no_weight_decay') else set()
        proj_kw = args.lr_linear_proj_names
        head = []
        backbone_decay = []
        backbone_no_decay = []
        linear_proj = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                if len(param.shape) == 1 or name.endswith(".bias") or name.split('.')[-1] in skip:
                    backbone_no_decay.append(param)
                else:
                    backbone_decay.append(param)
            elif match_name_keywords(name, proj_kw):
                linear_proj.append(param)
            else:
                head.append(param)
        param_dicts = [
            {"params": head, "lr": args.lr},
            {"params": backbone_decay, "lr": args.lr_backbone},
            {"params": backbone_no_decay, "lr": args.lr_backbone, "weight_decay": 0.},
            {"params": linear_proj, "lr": args.lr * args.lr_linear_proj_mult},
        ]

        # print the total number of trainable params.
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('num of total trainable prams:' + str(n_parameters))
        print(f'[optimizer] groups: head={len(head)} @ {args.lr:g} | '
              f'backbone={len(backbone_decay)+len(backbone_no_decay)} @ {args.lr_backbone:g} | '
              f'linear_proj={len(linear_proj)} @ {args.lr * args.lr_linear_proj_mult:g}')

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        return optimizer

    # build an optiimzer along with a learning scheduler
    optimizer = build_optimizer(model_without_ddp, args)
    

    # build data loader
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    print("# train:", len(dataset_train), ", # val", len(dataset_val))

    # data samplers
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    lr_scheduler, _ = create_scheduler(args, optimizer, num_steps=len(data_loader_train))

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = None

    output_dir = Path(args.output_dir)

    # resume from a checkpoint or eval with a checkpoint
    global_step = 0
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # Mid-epoch saves (epoch_completed=False) resume at the same epoch —
            # the per-epoch lr_scheduler.step(epoch) has not yet run for it.
            # End-of-epoch saves (or legacy ckpts without the flag) resume next.
            if checkpoint.get('epoch_completed', True):
                args.start_epoch = checkpoint['epoch'] + 1
            else:
                args.start_epoch = checkpoint['epoch']
            global_step = checkpoint.get('global_step', 0)
        print('load a checkpoint from', args.resume)

    # only evaluation purpose
    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device)

        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()

    def save_ckpt(path, epoch, global_step, epoch_completed):
        """Atomic, main-process-only full-state save (temp file + rename)."""
        if not utils.is_main_process():
            return
        path = Path(path)
        tmp = path.with_suffix('.pth.tmp')
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'epoch_completed': epoch_completed,
            'global_step': global_step,
            'args': args,
        }, tmp)
        tmp.replace(path)

    # mid-epoch save target: the rolling checkpoint.pth (epoch_completed=False)
    def save_cb(ep, step):
        save_ckpt(output_dir / 'checkpoint.pth', ep, step, epoch_completed=False)

    if not args.resume:
        args.start_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):

        # specify the current epoch number for samplers
        if args.distributed:
            sampler_train.set_epoch(epoch)

        if teacher_model is None:
            # training one epoch with default setting
            train_stats, global_step = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, n_iter_to_acc=args.n_iter_to_acc, print_freq=args.print_freq,
                global_step=global_step, wandb_run=wandb_run,
                ckpt_every=(args.ckpt_every_steps if args.output_dir else 0), save_cb=save_cb)
        else:
            # training one epoch with distillation with token matching
            train_stats, global_step = train_one_epoch_with_teacher(
                model, teacher_model, criterion, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, n_iter_to_acc=args.n_iter_to_acc, print_freq=args.print_freq,
                global_step=global_step, wandb_run=wandb_run,
                ckpt_every=(args.ckpt_every_steps if args.output_dir else 0), save_cb=save_cb)

        lr_scheduler.step(epoch)

        # model save (end of epoch -> epoch_completed=True)
        if args.output_dir:
            save_ckpt(output_dir / 'checkpoint.pth', epoch, global_step, epoch_completed=True)
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                save_ckpt(output_dir / f'checkpoint{epoch:04}.pth', epoch, global_step,
                          epoch_completed=True)

        # evaluation on COCO val.
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device)

        # logs
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if wandb_run is not None:
            # log learning rates for each param group
            lr_dict = {
                "lr/head": optimizer.param_groups[0]["lr"],
                "lr/backbone_no_decay": optimizer.param_groups[1]["lr"],
                "lr/backbone_decay": optimizer.param_groups[2]["lr"],
            }
            # No explicit step= : epoch is the step_metric for train_*/test_*/lr/*,
            # and mixing explicit steps with the per-step logs would be dropped by wandb.
            wandb.log({**log_stats, **lr_dict})

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    if wandb_run is not None:
        wandb.finish()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    parser = argparse.ArgumentParser('ViDT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    ''' for testing
    args.method = 'vidt'
    args.backbone_name = 'swin_tiny'
    args.batch_size = 2
    args.num_workers = 2
    args.aux_loss = True
    args.with_box_refine = True
    args.output_dir = 'testing'
    '''

    # set dim_feedforward differently
    # standard Transformers use 2048, while Deformable Transformers use 1024
    if args.method == 'vidt_wo_neck':
        args.dim_feedforward = 2048
    else:
        args.dim_feedforward = 1024

    # log file name
    if args.output_dir == '':
        # default out_dir name if not specified
        args.output_dir += args.method + '-'
        args.output_dir += args.backbone_name + '-'
        args.output_dir += args.sched + '-'
        args.output_dir += str(args.epochs) + '-'
        args.output_dir += str(args.batch_size)
        args.output_dir = args.method + '-' + args.backbone_name.upper() + '-batch-' + \
                          str(args.batch_size) + '-epoch-' + str(args.epochs)

    # make log directories
    if args.output_dir:
        log_main = 'logs'
        args.output_dir = os.path.join(log_main, args.output_dir)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print('log', args.output_dir)

    main(args)

