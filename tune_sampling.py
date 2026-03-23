#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tune_sampling.py
===================
Fast, one-shot tuner for patch-sampling hyper-parameters.

Strategy
--------
1. Load config + train scene list (same as quickstart.py).
2. Pre-load all scenes into RAM with the configured down_sample_scale
   (mirrors AI4ArcticChallengeDataset.__init__).
3. Pre-sample N "raw" patches applying only the *fixed* filters
   (valid_mask_threshold, sod_invalid_max_ratio).  Water-rejection and
   rare-class sampling are intentionally skipped here.
4. For every candidate parameter combination, compute the *expected* SOD
   pixel distribution analytically by weighting each raw patch with its
   acceptance probability under those parameters — no re-sampling needed.
5. Exhaustive grid search over the four tunable knobs; each evaluation is
   O(N_raw), so the full search completes in seconds.
6. Print the best parameter set as a ready-to-paste config snippet.

Usage
-----
    # with the server config (paths come from all.py):
    python tune_sampling.py configs/course_report/all.py

    # override data directory for a local run:
    python tune_sampling.py work_dirs/My_DS17/all.py \\
        --data-dir F:/ZJU/11_Ice/dataset_create/dataset_nc_zscore_new \\
        --train-list datalists/train_list.json \\
        --n-raw 3000
"""

from __future__ import annotations

import argparse
import builtins
import json
import os
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm


def get_variable_options(train_options: dict) -> dict:
    """Set variable category lists for MyDS without importing loaders.py."""
    train_options['sar_variables'] = list(train_options['train_variables'])
    train_options['full_variables'] = (
        list(train_options['charts']) + train_options['sar_variables']
    )
    train_options['amsrenv_variables'] = []
    train_options['auxiliary_variables'] = []
    return train_options


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Config loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_train_options_from_py(config_path: str) -> dict:
    """Load `train_options` from a Python config file without mmcv/import side effects."""
    abs_path = str(Path(config_path).resolve())
    namespace: dict = {
        '__builtins__': builtins.__dict__,
        # Some configs reference these symbols; they are irrelevant for this tuner.
        'f1_metric': None,
        'r2_metric': None,
    }

    with open(abs_path, 'r', encoding='utf-8') as f:
        src = f.read()

    # Avoid importing project-wide training dependencies (e.g. smp) from config files.
    filtered_lines = []
    for line in src.splitlines():
        if line.strip().startswith('from functions import'):
            continue
        filtered_lines.append(line)

    code = compile('\n'.join(filtered_lines), abs_path, 'exec')
    exec(code, namespace, namespace)

    if 'train_options' not in namespace:
        raise KeyError(f"Config file has no 'train_options': {config_path}")
    if not isinstance(namespace['train_options'], dict):
        raise TypeError(f"'train_options' must be a dict in {config_path}")

    return dict(namespace['train_options'])


def _load_config_with_base(config_path: str) -> dict:
    """Load train_options with `_base_` inheritance, without importing config modules."""
    cfg_path = Path(config_path).resolve()
    namespace: dict = {
        '__builtins__': builtins.__dict__,
        'f1_metric': None,
        'r2_metric': None,
    }

    with open(cfg_path, 'r', encoding='utf-8') as f:
        src = f.read()

    filtered_lines = []
    for line in src.splitlines():
        if line.strip().startswith('from functions import'):
            continue
        filtered_lines.append(line)

        code = compile('\n'.join(filtered_lines), str(cfg_path), 'exec')
        exec(code, namespace, namespace)

    child_opts = namespace.get('train_options', {})
    if not isinstance(child_opts, dict):
        raise TypeError(f"'train_options' must be a dict in {config_path}")

    merged_opts: dict = {}
    base_list = namespace.get('_base_', [])
    if isinstance(base_list, str):
        base_list = [base_list]

    for base_rel in base_list:
        base_path = (cfg_path.parent / base_rel).resolve()
        base_opts = _load_train_options_from_py(str(base_path))
        for k, v in base_opts.items():
            merged_opts.setdefault(k, v)

    # Preserve original behavior: child sets top-level keys; missing keys come from base.
    merged_opts.update(dict(child_opts))

    return merged_opts


def open_nc_dataset(nc_path: str):
    """Open one NetCDF file with backend fallbacks."""
    engines = ['h5netcdf', 'netcdf4', None]
    last_err: Exception | None = None

    for engine in engines:
        try:
            if engine is None:
                return xr.open_dataset(nc_path, mask_and_scale=False)
            return xr.open_dataset(nc_path, engine=engine, mask_and_scale=False)
        except ModuleNotFoundError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        "Failed to open NetCDF file with available engines "
        "(tried: h5netcdf, netcdf4, default). "
        "Please install one backend, e.g. 'pip install h5netcdf' or "
        "'pip install netCDF4'."
    ) from last_err

def load_options(config_path: str, data_dir: str | None = None) -> dict:
    """Load train_options from a Python config file, merged with base defaults."""
    opts = _load_config_with_base(config_path)

    if data_dir:
        opts['path_to_train_data'] = data_dir

    opts = get_variable_options(opts)
    return opts


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Scene pre-loading  (mirrors AI4ArcticChallengeDataset.__init__)
# ─────────────────────────────────────────────────────────────────────────────

def load_scenes(opts: dict, files: list[str]) -> list[torch.Tensor]:
    """Downsample and load every training scene into RAM as a (C, H, W) tensor."""
    scale = opts['down_sample_scale']
    ps    = opts['patch_size']

    print(f"\nLoading {len(files)} scenes  (down_sample_scale={scale}) …")
    scenes: list[torch.Tensor] = []

    for fname in tqdm(files):
        path = os.path.join(opts['path_to_train_data'], fname)
        ds = open_nc_dataset(path)

        arr = ds[opts['full_variables']].to_array().values       # (C, H, W)
        t   = torch.from_numpy(arr[np.newaxis]).float()           # (1, C, H, W)

        if scale != 1:
            t = torch.nn.functional.interpolate(
                t,
                size=(t.size(2) // scale, t.size(3) // scale),
                mode=opts['loader_downsampling'],
            )

        # Pad scenes that are smaller than patch_size (same logic as loaders.py).
        hp = max(0, ps - t.size(2) + 1)
        wp = max(0, ps - t.size(3) + 1)
        if hp or wp:
            nc  = len(opts['charts'])
            y_t = torch.nn.functional.pad(t[:, :nc], (0, wp, 0, hp), value=255)
            x_t = torch.nn.functional.pad(t[:, nc:], (0, wp, 0, hp), value=0)
            t   = torch.cat([y_t, x_t], dim=1)

        scenes.append(t.squeeze(0))   # (C, H, W)
        ds.close()

    return scenes


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Raw-patch pre-sampling  (fixed filters only)
# ─────────────────────────────────────────────────────────────────────────────

def presample_raw(
    scenes: list[torch.Tensor],
    opts:   dict,
    n:      int,
    seed:   int = 42,
) -> list[dict]:
    """
    Sample `n` patches using only the fixed filters:
      • valid_mask_threshold   — discard black-border patches
      • sod_invalid_max_ratio  — discard patches with too many unlabelled pixels

    Water-rejection and rare-class sampling are deliberately NOT applied here;
    their effect will be modelled analytically in expected_dist().

    Each returned record contains:
      cls_counts : ndarray(n_cls+1,) int64  — pixel counts for cls 0…(n_cls-1)
                                              and mask(255) at index n_cls
      water_frac : float  — fraction of valid pixels that are water (cls 0)
      cls_fracs  : ndarray(n_cls,) float32  — fraction of valid pixels per class
    """
    rng      = np.random.default_rng(seed)
    sod_idx  = opts['charts'].index('SOD')
    fill     = opts['class_fill_values']['SOD']    # 255
    ps       = opts['patch_size']
    n_cls    = opts['n_classes']['SOD']
    vm_thr   = opts.get('valid_mask_threshold', 0.5)
    inv_max  = opts.get('sod_invalid_max_ratio', 1.0)

    gvm_idx: int | None = None
    if 'global_valid_mask' in opts['train_variables']:
        gvm_idx = (len(opts['charts']) +
                   list(opts['train_variables']).index('global_valid_mask'))

    records: list[dict] = []
    attempts = 0
    max_att  = n * 500
    pbar = tqdm(total=n, desc="Pre-sampling raw patches")

    while len(records) < n and attempts < max_att:
        attempts += 1
        sid = int(rng.integers(0, len(scenes)))
        sc  = scenes[sid]
        H, W = sc.size(1), sc.size(2)
        if H <= ps or W <= ps:
            continue
        r = int(rng.integers(0, H - ps))
        c = int(rng.integers(0, W - ps))

        # — Filter: global_valid_mask (black-border check) —
        if gvm_idx is not None:
            vm = sc[gvm_idx, r:r + ps, c:c + ps].numpy()
            if float(vm.mean()) < vm_thr:
                continue

        # — Extract SOD patch and apply the standard label remap —
        sod = sc[sod_idx, r:r + ps, c:c + ps].numpy().flatten().copy()
        valid = sod != fill
        sod[valid & (sod >= 2)] -= 1   # merge: old 2→1, 3→2, 4→3, 5→4

        if int(valid.sum()) <= 1:
            continue

        # — Filter: sod_invalid_max_ratio —
        if inv_max < 1.0:
            if float((~valid).sum()) / len(sod) > inv_max:
                continue

        valid_sod = sod[valid]
        n_valid   = len(valid_sod)

        cls_counts = np.array(
            [(sod == cls).sum() for cls in range(n_cls)] + [int((~valid).sum())],
            dtype=np.int64,
        )
        cls_fracs = np.array(
            [(valid_sod == cls).sum() / n_valid for cls in range(n_cls)],
            dtype=np.float32,
        )

        records.append({
            'cls_counts': cls_counts,
            'water_frac': float(cls_fracs[0]),
            'cls_fracs':  cls_fracs,
        })
        pbar.update(1)

    pbar.close()

    if len(records) < n:
        print(f"  ⚠  Only collected {len(records)}/{n} patches "
              f"after {attempts} attempts.")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Analytical expected distribution under a parameter set
# ─────────────────────────────────────────────────────────────────────────────

def expected_dist(
    records:  list[dict],
    params:   dict,
    n_cls:    int = 5,
) -> np.ndarray:
    """
    Compute the expected SOD pixel-count vector for the given sampling
    parameters using the pre-sampled raw patches.

    For each raw patch p, the acceptance weight is:
        w(p) = water_accept(p) × rare_accept(p)

    where:
        water_accept(p) = 1 − (water_frac_p > water_max) × water_prob
        rare_accept(p)  = (1 − alpha) + alpha × Σ_{c ∈ rare_cls} cls_fracs_p[c]

    Expected counts = Σ_p  w(p) × cls_counts(p)

    Returns ndarray of shape (n_cls+1,): classes 0…n_cls-1 plus mask.
    """
    w_max  = float(params.get('water_patch_max_ratio', 1.0))
    w_prob = float(params.get('water_rejection_prob',  0.0))
    rare   = params.get('rare_sampling_classes', [])
    alpha  = float(params.get('rare_sampling_alpha',   0.0))

    totals = np.zeros(n_cls + 1, dtype=np.float64)

    for rec in records:
        # water-rejection acceptance probability
        water_accept = 1.0 - (float(rec['water_frac'] > w_max) * w_prob)

        # rare-class acceptance probability
        if rare and alpha > 0.0:
            rare_frac    = float(sum(rec['cls_fracs'][c] for c in rare))
            rare_accept  = (1.0 - alpha) + alpha * rare_frac
        else:
            rare_accept  = 1.0

        totals += water_accept * rare_accept * rec['cls_counts'].astype(np.float64)

    return totals


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Balance metrics and display
# ─────────────────────────────────────────────────────────────────────────────

def coeff_of_variation(dist: np.ndarray, n_cls: int = 5) -> float:
    """CV = std/mean across the n_cls SOD classes (lower = more balanced)."""
    c = dist[:n_cls].astype(float)
    m = c.mean()
    return float(c.std() / m) if m > 0 else float('inf')


def max_min_ratio(dist: np.ndarray, n_cls: int = 5) -> float:
    c = dist[:n_cls].astype(float)
    mn = c.min()
    return float(c.max() / mn) if mn > 0 else float('inf')


def print_dist(dist: np.ndarray, n_cls: int = 5, label: str = ""):
    total_valid = float(dist[:n_cls].sum())
    total_all   = total_valid + float(dist[n_cls])

    print(f"\n  ── {label} ──")
    for c in range(n_cls):
        frac = dist[c] / total_valid if total_valid > 0 else 0.0
        bar  = '█' * int(frac * 36)
        print(f"    cls {c}: {frac * 100:5.1f}%  {bar}")
    mask_frac = dist[n_cls] / total_all if total_all > 0 else 0.0
    print(f"    mask:  {mask_frac * 100:5.1f}%")
    print(f"    CV={coeff_of_variation(dist, n_cls):.4f}   "
          f"max/min={max_min_ratio(dist, n_cls):.2f}×")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Exhaustive grid search
# ─────────────────────────────────────────────────────────────────────────────

# Search grids for each knob.
_WATER_MAX_GRID  = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]
_WATER_PROB_GRID = [0.00, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98]
_ALPHA_GRID      = [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
_RARE_CLS_GRID   = [
    [],
    [1], [2], [4],
    [1, 2], [1, 4], [2, 4],
    [1, 2, 4],
    [1, 2, 3, 4],
]


def grid_search(
    records:     list[dict],
    base_params: dict,
    n_cls:       int = 5,
) -> tuple[dict, np.ndarray]:
    """
    Find the parameter combination minimising CV across SOD classes.
    Each candidate is evaluated analytically — no re-sampling needed.
    """
    # Prune combinations that are trivially equivalent:
    #   alpha=0  with any rare_cls  →  same as alpha=0, rare_cls=[]
    #   rare_cls=[]  with alpha>0   →  rare filter inactive
    combos = [
        (w_max, w_prob, alpha, rare_cls)
        for w_max, w_prob, alpha, rare_cls
        in product(_WATER_MAX_GRID, _WATER_PROB_GRID, _ALPHA_GRID, _RARE_CLS_GRID)
        if not (alpha == 0.0 and rare_cls)     # deduplicate alpha=0 cases
        and not (alpha > 0.0 and not rare_cls) # rare_cls must be set if alpha>0
    ]

    print(f"\nGrid search over {len(combos):,} combinations …")

    best_cv   = float('inf')
    best_p    = deepcopy(base_params)
    best_dist = expected_dist(records, base_params, n_cls)
    inv_max   = base_params.get('sod_invalid_max_ratio', 1.0)

    for w_max, w_prob, alpha, rare_cls in tqdm(combos, desc="Searching"):
        params = {
            'sod_invalid_max_ratio': inv_max,
            'water_patch_max_ratio': w_max,
            'water_rejection_prob':  w_prob,
            'rare_sampling_classes': list(rare_cls),
            'rare_sampling_alpha':   alpha,
        }
        d  = expected_dist(records, params, n_cls)
        cv = coeff_of_variation(d, n_cls)
        if cv < best_cv:
            best_cv   = cv
            best_p    = deepcopy(params)
            best_dist = d.copy()

    return best_p, best_dist


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Output
# ─────────────────────────────────────────────────────────────────────────────

def print_config_snippet(params: dict):
    rare = params['rare_sampling_classes']
    sep  = "=" * 60
    print(f"\n{sep}")
    print("  Recommended config snippet  — paste into all.py")
    print(sep)
    print(f"    'sod_invalid_max_ratio': {params['sod_invalid_max_ratio']},")
    print(f"    'water_patch_max_ratio': {params['water_patch_max_ratio']},")
    print(f"    'water_rejection_prob':  {params['water_rejection_prob']},")
    print(f"    'rare_sampling_classes': {rare},")
    print(f"    'rare_sampling_alpha':   {params['rare_sampling_alpha']},")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tune patch-sampling parameters for balanced SOD distribution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'config',
        help='Path to train config, e.g. configs/course_report/all.py',
    )
    parser.add_argument(
        '--data-dir', default=None,
        help='Override path_to_train_data (useful for local runs where '
             'the server path in the config does not exist)',
    )
    parser.add_argument(
        '--train-list', default=None,
        help='Override train_list_path in config',
    )
    parser.add_argument(
        '--n-raw', type=int, default=2000,
        help='Number of raw patches to pre-sample (default: 2000). '
             'More → slower pre-sample, more accurate search.',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for pre-sampling (default: 42)',
    )
    parser.add_argument(
        '--target-cv', type=float, default=0.35,
        help='CV target for "balanced" (default: 0.35). The best found '
             'parameters are reported regardless.',
    )
    args = parser.parse_args()

    # ── load config ───────────────────────────────────────────────────────────
    print(f"Config: {args.config}")
    opts = load_options(args.config, args.data_dir)

    if args.train_list:
        opts['train_list_path'] = args.train_list

    with open(opts['train_list_path']) as f:
        files: list[str] = json.load(f)

    n_cls = opts['n_classes']['SOD']

    print(f"Train list : {opts['train_list_path']}  ({len(files)} scenes)")
    print(f"Data dir   : {opts['path_to_train_data']}")
    print(f"Patch size : {opts['patch_size']}  |  "
          f"Down-sample : ×{opts['down_sample_scale']}")
    print(f"n_classes  : {n_cls}")

    # ── collect current sampling parameters ───────────────────────────────────
    base_params = {
        'sod_invalid_max_ratio': opts.get('sod_invalid_max_ratio', 1.0),
        'water_patch_max_ratio': opts.get('water_patch_max_ratio', 1.0),
        'water_rejection_prob':  opts.get('water_rejection_prob',  0.0),
        'rare_sampling_classes': list(opts.get('rare_sampling_classes', [])),
        'rare_sampling_alpha':   opts.get('rare_sampling_alpha',   0.0),
    }
    print("\nCurrent sampling parameters:")
    for k, v in base_params.items():
        print(f"  {k}: {v}")

    # ── pre-load scenes ───────────────────────────────────────────────────────
    scenes = load_scenes(opts, files)

    # ── pre-sample raw patches ────────────────────────────────────────────────
    records = presample_raw(scenes, opts, n=args.n_raw, seed=args.seed)
    print(f"\nPre-sampled {len(records)} raw patches.")

    # ── baseline distribution ─────────────────────────────────────────────────
    baseline = expected_dist(records, base_params, n_cls)
    print_dist(baseline, n_cls, label="Baseline  (current config)")

    # ── grid search ───────────────────────────────────────────────────────────
    best_params, best_dist = grid_search(records, base_params, n_cls)
    best_cv_val = coeff_of_variation(best_dist, n_cls)

    status = "✓ target met" if best_cv_val <= args.target_cv else f"target={args.target_cv}"
    print_dist(best_dist, n_cls,
               label=f"Best found  (CV={best_cv_val:.4f}  {status})")

    print_config_snippet(best_params)

    # ── advisory when target not met ─────────────────────────────────────────
    if best_cv_val > args.target_cv:
        dominant = int(np.argmax(best_dist[:n_cls]))
        print(f"\n⚠  Best CV={best_cv_val:.4f} still exceeds target={args.target_cv}.")
        print(f"   Dominant class after tuning: cls {dominant}")
        print("   The four tunable knobs cannot fully balance the dataset.")
        print("   Consider one or more of:")
        print("   • Remove scenes that are overwhelmingly class 3 (thick FY ice)")
        print("     — check which scenes contribute most class-3 pixels.")
        print("   • Add a 'common class rejection' filter for class 3 (analogous")
        print("     to the existing water-rejection filter).")
        print("   • Accept the imbalance and compensate with higher loss weights")
        print("     for under-represented classes in chart_loss['SOD']['weight'].")
    else:
        print(f"\n✓  Distribution balanced to CV ≤ {args.target_cv}.")


if __name__ == '__main__':
    main()
