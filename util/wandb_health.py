"""Reusable run-health instrumentation (drop-in for any clip_ddetr / sketch_detr / locformer run).

Three things, all robust + uniform across branches:
  1. log_run_provenance(run, repo_dir) -> git commit/branch/dirty into wandb.config AND summary,
     so the version is ALWAYS visible in the runs table (don't rely on wandb's flaky auto-capture).
  2. wandb_save_checkpoints(run, output_dir) -> upload ONLY best_AP.pth + checkpoint.pth, overwriting
     each epoch (bounded ~2 files/run).
  3. log_grad_health(model, step, run, groups) -> per-component grad L2 norm as wandb scalars +
     a bar chart + a NaN-grad count. Call AFTER backward(), BEFORE optimizer.zero_grad().
"""
import os
import subprocess

try:
    import wandb
except Exception:
    wandb = None


def _git(args, cwd):
    try:
        return subprocess.check_output(['git'] + args, cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def log_run_provenance(run, repo_dir=None):
    """Log git commit/branch/dirty to wandb.config + summary. repo_dir defaults to this file's repo."""
    if run is None or wandb is None:
        return
    cwd = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    short = _git(['rev-parse', '--short', 'HEAD'], cwd)
    info = {
        'git_commit': _git(['rev-parse', 'HEAD'], cwd),
        'git_commit_short': short,
        'git_branch': _git(['rev-parse', '--abbrev-ref', 'HEAD'], cwd),
        'git_dirty': bool(_git(['status', '--porcelain'], cwd)),
    }
    try:
        run.config.update(info, allow_val_change=True)
        run.summary['git_commit'] = short  # so it shows in the runs table column
    except Exception:
        pass


def wandb_save_checkpoints(run, output_dir, names=('checkpoint.pth', 'best_AP.pth')):
    """Upload latest + best_AP checkpoint, overwriting each epoch (bounded storage)."""
    if run is None or wandb is None:
        return
    for f in names:
        p = os.path.join(output_dir, f)
        if os.path.exists(p):
            try:
                wandb.save(p, base_path=str(output_dir), policy='now')
            except Exception:
                pass


# param-name -> component bucket (override per repo via `groups`)
DEFAULT_GROUPS = {
    'backbone_img':    ('backbone.0', 'backbone'),
    'backbone_sketch': ('backbone_sketch',),
    'transformer':     ('transformer',),
    'fusion':          ('fusion', 'de_path', 'depath', 'film', 'token_scorer', 'sketch_pool'),
    'input_proj':      ('input_proj',),
    'heads':           ('class_embed', 'bbox_embed'),
    'query':           ('query_embed', 'query_feat', 'tgt'),
}


def _bucket(name, groups):
    for g, prefixes in groups.items():
        for p in prefixes:
            if name.startswith(p) or ('.' + p) in name:
                return g
    return 'other'


def log_grad_health(model, step, run, groups=None, prefix='grad'):
    """Per-component grad L2 norm -> wandb scalars + bar chart + NaN count.
    Call AFTER backward(), BEFORE zero_grad(). Cheap; gate the call by step%freq at the call site."""
    if run is None or wandb is None:
        return
    groups = groups or DEFAULT_GROUPS
    sq = {}
    nan = 0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not bool((g == g).all()):
            nan += 1
        comp = _bucket(n, groups)
        sq[comp] = sq.get(comp, 0.0) + float(g.norm(2).item()) ** 2
    norms = {c: v ** 0.5 for c, v in sq.items()}
    log = {f'{prefix}/{c}': v for c, v in norms.items()}
    log[f'{prefix}/total'] = sum(sq.values()) ** 0.5
    log[f'{prefix}/nan_param_tensors'] = nan
    try:
        tbl = wandb.Table(data=[[c, v] for c, v in sorted(norms.items())],
                          columns=['component', 'grad_norm'])
        log[f'{prefix}/by_component_bar'] = wandb.plot.bar(tbl, 'component', 'grad_norm',
                                                           title='grad L2 norm per component')
    except Exception:
        pass
    try:
        run.log(log, step=step)
    except Exception:
        pass
