#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scan_cls2.py
===============
扫描数据目录中所有 NC 文件，按 cls2（薄一年冰）像素占比从高到低排列，
帮助筛选含有大量 cls2 的场景，用于补充训练数据。

NC 文件内 SOD 原始值与训练类别的对应关系
-----------------------------------------
NC raw 0  → training cls 0  水体
NC raw 1  → training cls 1  新冰 (81,82)
NC raw 2  → training cls 1  幼冰/灰冰 (83,84,85)，由 loaders.py 合并
NC raw 3  → training cls 2  薄一年冰 (87,88,89)       ← 本脚本目标
NC raw 4  → training cls 3  中厚一年冰 (86,91,93)
NC raw 5  → training cls 4  多年冰 (95,96,97)
NC raw 255               无标签（mask）

Usage
-----
    # 扫描整个数据目录（包含 val/train 所有 NC 文件）：
    python scan_cls2.py --data-dir /root/autodl-tmp/My_dataset/

    # 扫描一个 JSON 列表中的文件（默认行为）：
    python scan_cls2.py --data-dir /root/autodl-tmp/My_dataset/ \\
                        --file-list datalists/train_list.json

    # 同时扫描多个 JSON 列表（自动去重）：
    python scan_cls2.py --data-dir /root/autodl-tmp/My_dataset/ \\
                        --file-list datalists/train_list.json datalists/val_list.json

    # 扫描目录内所有 NC 文件（不依赖 JSON）：
    python scan_cls2.py --data-dir /root/autodl-tmp/My_dataset/ --scan-dir

    # 保存结果为 CSV：
    python scan_cls2.py --data-dir /root/autodl-tmp/My_dataset/ \\
                        --file-list datalists/train_list.json --output cls2_ranking.csv

    # 只打印 cls2 占比 >= 5% 的场景：
    python scan_cls2.py --data-dir /root/autodl-tmp/My_dataset/ \\
                        --file-list datalists/train_list.json --min-ratio 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import xarray as xr
from tqdm import tqdm


# ---------------------------------------------------------------------------
# NC raw SOD value that maps to training cls 2 (thin FY ice)
# After loaders.py remap: raw >= 2  →  raw - 1
# So training cls 2  =  NC raw 3
# ---------------------------------------------------------------------------
CLS2_RAW_VALUE = 3

# All valid SOD raw values (excluding mask=255) in the NC files
ALL_RAW_CLASSES = [0, 1, 2, 3, 4, 5]
RAW_LABEL_NAMES = {
    0: 'water',
    1: 'new ice (81,82)',
    2: 'young/grey ice (83-85) → merged→cls1',
    3: 'thin FY ice (87-89)   → cls2  ★',
    4: 'medium/thick FY (86,91,93) → cls3',
    5: 'old/MY ice (95-97)    → cls4',
}


def open_nc_dataset(nc_path: str):
    """Open a NetCDF file with fallback engines.

    Try h5netcdf first (fast/common for h5-based NC), then netcdf4,
    and finally xarray default engine auto-detection.
    """
    engines = ['h5netcdf', 'netcdf4', None]
    last_err: Exception | None = None

    for engine in engines:
        try:
            if engine is None:
                return xr.open_dataset(nc_path, mask_and_scale=False)
            return xr.open_dataset(nc_path, engine=engine, mask_and_scale=False)
        except ModuleNotFoundError as e:
            # Missing optional backend dependency, try the next engine.
            last_err = e
            continue
        except Exception as e:
            # Engine exists but failed for other reasons; keep trying fallback.
            last_err = e
            continue

    raise RuntimeError(
        "Failed to open NetCDF file with available engines "
        "(tried: h5netcdf, netcdf4, default). "
        "Please install one backend, e.g. `pip install h5netcdf` or "
        "`pip install netCDF4`."
    ) from last_err


def scan_file(
    nc_path: str,
    sod_var: str = 'SOD',
    mask_var: str = 'global_valid_mask',
) -> dict | None:
    """
    Load one NC file and compute SOD class statistics.

    Returns a dict with:
        name        : filename (no directory)
        valid_px    : total valid SOD pixels (non-255, inside global_valid_mask)
        cls2_px     : pixels with NC raw SOD == CLS2_RAW_VALUE
        cls2_ratio  : cls2_px / valid_px  (0 if valid_px == 0)
        raw_counts  : dict {raw_val: count} for all SOD values
        scene_size  : (H, W) original scene dimensions
    """
    try:
        ds = open_nc_dataset(nc_path)
    except Exception as e:
        print(f"  ⚠  Cannot open {os.path.basename(nc_path)}: {e}", file=sys.stderr)
        return None

    if sod_var not in ds:
        print(f"  ⚠  '{sod_var}' not found in {os.path.basename(nc_path)}", file=sys.stderr)
        ds.close()
        return None

    sod = ds[sod_var].values.flatten().astype(np.int32)   # (H*W,)
    scene_size = ds[sod_var].values.shape

    # build valid pixel mask: SOD != 255  AND  global_valid_mask == 1 (if present)
    valid = sod != 255
    if mask_var in ds:
        gvm = ds[mask_var].values.flatten().astype(np.float32)
        valid = valid & (gvm > 0.5)

    ds.close()

    valid_px  = int(valid.sum())
    cls2_px   = int(((sod == CLS2_RAW_VALUE) & valid).sum())
    cls2_ratio = cls2_px / valid_px if valid_px > 0 else 0.0

    raw_counts = {v: int(((sod == v) & valid).sum()) for v in ALL_RAW_CLASSES}

    return {
        'name':       os.path.basename(nc_path),
        'valid_px':   valid_px,
        'cls2_px':    cls2_px,
        'cls2_ratio': cls2_ratio,
        'raw_counts': raw_counts,
        'scene_size': scene_size,
    }


def format_bar(ratio: float, width: int = 30) -> str:
    return '█' * int(ratio * width)


def print_results(
    results:   list[dict],
    top_n:     int  = 0,
    min_ratio: float = 0.0,
):
    filtered = [r for r in results if r['cls2_ratio'] >= min_ratio]
    if top_n > 0:
        filtered = filtered[:top_n]

    if not filtered:
        print("No scenes match the filter criteria.")
        return

    # header
    print(f"\n{'Rank':>4}  {'cls2%':>6}  {'cls2_px':>9}  {'valid_px':>9}  "
          f"{'H×W':>11}  {'Scene name'}")
    print("─" * 110)

    for rank, r in enumerate(filtered, 1):
        pct  = r['cls2_ratio'] * 100
        bar  = format_bar(r['cls2_ratio'], width=20)
        H, W = r['scene_size']
        print(f"{rank:>4}  {pct:>5.1f}%  {r['cls2_px']:>9,}  "
              f"{r['valid_px']:>9,}  {H:>5}×{W:<5}  {r['name']}")
        print(f"       {bar}")

    print(f"\n  Showing {len(filtered)} / {len(results)} scanned scenes"
          + (f"  (min_ratio={min_ratio*100:.1f}%)" if min_ratio > 0 else ""))

    # --- aggregate class distribution across all scanned scenes ---
    print("\n── Distribution across ALL scanned scenes ──")
    total_valid = sum(r['valid_px'] for r in results)
    total_counts = {v: sum(r['raw_counts'][v] for r in results) for v in ALL_RAW_CLASSES}
    for v in ALL_RAW_CLASSES:
        frac = total_counts[v] / total_valid if total_valid > 0 else 0.0
        bar  = format_bar(frac, width=24)
        marker = ' ←' if v == CLS2_RAW_VALUE else ''
        print(f"  raw {v}  {frac*100:5.1f}%  {bar}  {RAW_LABEL_NAMES[v]}{marker}")
    print(f"  (total valid pixels: {total_valid:,})")


def save_csv(results: list[dict], output_path: str):
    fieldnames = ['rank', 'name', 'cls2_ratio_pct', 'cls2_px', 'valid_px',
                  'scene_H', 'scene_W'] + [f'raw{v}_px' for v in ALL_RAW_CLASSES]
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, r in enumerate(results, 1):
            H, W = r['scene_size']
            row = {
                'rank':           rank,
                'name':           r['name'],
                'cls2_ratio_pct': round(r['cls2_ratio'] * 100, 2),
                'cls2_px':        r['cls2_px'],
                'valid_px':       r['valid_px'],
                'scene_H':        H,
                'scene_W':        W,
            }
            for v in ALL_RAW_CLASSES:
                row[f'raw{v}_px'] = r['raw_counts'][v]
            writer.writerow(row)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Rank NC scenes by cls2 (thin FY ice) pixel fraction.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--data-dir', required=True,
        help='Directory containing the NC dataset files.',
    )
    parser.add_argument(
        '--file-list', nargs='+', default=None, metavar='JSON',
        help='One or more JSON list files to scan (e.g. datalists/train_list.json '
             'datalists/val_list.json). Files are deduplicated. '
             'Required unless --scan-dir is used.',
    )
    parser.add_argument(
        '--scan-dir', action='store_true',
        help='Scan ALL .nc files in data-dir instead of using --file-list.',
    )
    parser.add_argument(
        '--sod-var', default='SOD',
        help='Name of SOD variable in NC files (default: SOD).',
    )
    parser.add_argument(
        '--mask-var', default='global_valid_mask',
        help='Name of valid-mask variable (default: global_valid_mask). '
             'Set to "" to disable.',
    )
    parser.add_argument(
        '--top', type=int, default=30,
        help='Print top N scenes by cls2 ratio (default: 30; 0 = print all).',
    )
    parser.add_argument(
        '--min-ratio', type=float, default=0.0,
        help='Only show scenes with cls2 ratio >= this value (e.g. 0.05 for 5%%).',
    )
    parser.add_argument(
        '--output', default=None,
        help='Optional path to save full ranking as CSV.',
    )
    parser.add_argument(
        '--verify-sample', action='store_true',
        help='Print raw SOD value distribution for the first NC file, then exit. '
             'Use this to confirm the CLS2_RAW_VALUE mapping is correct.',
    )
    args = parser.parse_args()

    mask_var = args.mask_var if args.mask_var else None

    # ── collect file list ────────────────────────────────────────────────────
    if args.scan_dir:
        files = sorted(
            os.path.join(args.data_dir, fn)
            for fn in os.listdir(args.data_dir)
            if fn.lower().endswith('.nc')
        )
        print(f"Scanning ALL {len(files)} NC files in {args.data_dir}")
    elif args.file_list:
        # merge multiple JSON lists, preserve order, deduplicate
        seen: set[str] = set()
        filenames: list[str] = []
        for json_path in args.file_list:
            with open(json_path, encoding='utf-8') as f:
                for fn in json.load(f):
                    if fn not in seen:
                        seen.add(fn)
                        filenames.append(fn)
        files = [os.path.join(args.data_dir, fn) for fn in filenames]
        src = ', '.join(args.file_list)
        print(f"Scanning {len(files)} files from: {src}")
    else:
        parser.error("Specify --file-list <json> [<json> ...] or use --scan-dir.")

    if not files:
        print("No NC files found. Check --data-dir.")
        return

    # ── verify mode: inspect first file and exit ─────────────────────────────
    if args.verify_sample:
        path = files[0]
        print(f"\nVerifying SOD value distribution in: {os.path.basename(path)}")
        ds = open_nc_dataset(path)
        sod = ds[args.sod_var].values.flatten().astype(np.int32)
        ds.close()
        vals, cnts = np.unique(sod, return_counts=True)
        total = len(sod)
        print(f"{'SOD raw value':>15}  {'count':>10}  {'%':>6}  label")
        print("─" * 70)
        for v, c in zip(vals, cnts):
            marker = ' ← target cls2' if v == CLS2_RAW_VALUE else ''
            label  = RAW_LABEL_NAMES.get(int(v), '(mask/other)')
            print(f"{v:>15}  {c:>10,}  {c/total*100:>5.1f}%  {label}{marker}")
        return

    # ── scan all files ────────────────────────────────────────────────────────
    results = []
    for path in tqdm(files, desc="Scanning NC files"):
        r = scan_file(path, sod_var=args.sod_var,
                      mask_var=mask_var or '')
        if r is not None:
            results.append(r)

    if not results:
        print("No valid results.")
        return

    # sort by cls2_ratio descending
    results.sort(key=lambda x: x['cls2_ratio'], reverse=True)

    # ── print ─────────────────────────────────────────────────────────────────
    print_results(results, top_n=args.top, min_ratio=args.min_ratio)

    # ── save CSV ──────────────────────────────────────────────────────────────
    if args.output:
        save_csv(results, args.output)


if __name__ == '__main__':
    main()
