#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""check_val_repr.py
====================
检验验证集相对于训练集的代表性，重点关注 cls2/cls3 分布差异。

SOD 原始值 → 训练类别映射（与 loaders.py / scan_cls2.py 一致）
--------------------------------------------------------------
NC raw 0  → cls0  水体
NC raw 1  → cls1  新冰
NC raw 2  → cls1  幼冰/灰冰（合并）
NC raw 3  → cls2  薄一年冰  ★ 问题类别
NC raw 4  → cls3  中厚一年冰
NC raw 5  → cls4  多年冰
NC raw 255       mask（无标签）

输出内容
--------
1. 逐场景类别分布表（训练集 / 验证集分开）
2. 训练集 vs 验证集整体类别分布对比
3. KL 散度（衡量分布差异）
4. 每个验证场景 cls2 覆盖率的异常检测
5. 季节分布对比

用法示例
--------
    python check_val_repr.py \\
        --data-dir /root/autodl-tmp/My_dataset/ \\
        --train-list datalists/train_list_small.json \\
        --val-list datalists/val_list2.json

    # 同时比较两个验证集：
    python check_val_repr.py \\
        --data-dir /root/autodl-tmp/My_dataset/ \\
        --train-list datalists/train_list_small.json \\
        --val-list datalists/val_list2.json datalists/val_list1.json \\
        --output repr_report.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import xarray as xr
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 常量：SOD 原始值 → 训练类别
# ---------------------------------------------------------------------------
RAW_TO_TRAIN = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4}
N_TRAIN_CLS = 5
TRAIN_CLS_NAMES = {
    0: 'cls0 water',
    1: 'cls1 new/young ice',
    2: 'cls2 thin FYI  ★',
    3: 'cls3 med/thick FYI',
    4: 'cls4 MY ice',
}
MASK_VAL = 255


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def open_nc(path: str):
    for engine in ('h5netcdf', 'netcdf4', None):
        try:
            kw = {'engine': engine} if engine else {}
            return xr.open_dataset(path, mask_and_scale=False, **kw)
        except Exception:
            continue
    raise RuntimeError(f'Cannot open {path}')


def scene_class_dist(nc_path: str, sod_var: str = 'SOD',
                     mask_var: str = 'global_valid_mask') -> dict | None:
    """返回一个场景的训练类别像素数字典，以及场景基本信息。"""
    try:
        ds = open_nc(nc_path)
    except Exception as e:
        print(f'  [WARN] {os.path.basename(nc_path)}: {e}', file=sys.stderr)
        return None

    if sod_var not in ds:
        ds.close()
        return None

    sod = ds[sod_var].values.flatten().astype(np.int32)
    valid = sod != MASK_VAL
    if mask_var in ds:
        gvm = ds[mask_var].values.flatten().astype(np.float32)
        valid &= gvm > 0.5
    ds.close()

    train_counts = {c: 0 for c in range(N_TRAIN_CLS)}
    for raw, tcls in RAW_TO_TRAIN.items():
        train_counts[tcls] += int(((sod == raw) & valid).sum())

    valid_px = int(valid.sum())
    name = os.path.basename(nc_path)
    # 从文件名提取月份：S1A_EW_GRDM_1SDH_YYYYMMDDTHHMMSS_...
    try:
        month = int(name[21:23])
    except (ValueError, IndexError):
        month = -1

    return {
        'name': name,
        'valid_px': valid_px,
        'counts': train_counts,          # {cls: pixel_count}
        'ratios': {c: train_counts[c] / valid_px if valid_px > 0 else 0.0
                   for c in range(N_TRAIN_CLS)},
        'month': month,
    }


def load_json_list(path: str) -> list[str]:
    with open(path, encoding='utf-8') as f:
        return [fn for fn in json.load(f) if isinstance(fn, str)]


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """KL(P || Q)，p=train分布, q=val分布。值越小越相似。"""
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon 散度，对称且有界 [0,1]（用 log2 时最大为 1 bit）。"""
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    p /= p.sum() + 1e-9
    q /= q.sum() + 1e-9
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def bar(ratio: float, width: int = 20) -> str:
    return '█' * int(ratio * width) + '░' * (width - int(ratio * width))


SEASON = {12: 'Winter', 1: 'Winter', 2: 'Winter',
           3: 'Spring', 4: 'Spring', 5: 'Spring',
           6: 'Summer', 7: 'Summer', 8: 'Summer',
           9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}


# ---------------------------------------------------------------------------
# 报告函数
# ---------------------------------------------------------------------------

def print_scene_table(records: list[dict], title: str):
    print(f'\n{"─"*100}')
    print(f'  {title}  （共 {len(records)} 个场景）')
    print(f'{"─"*100}')
    hdr = f'  {"Scene":<55} {"cls0":>6} {"cls1":>6} {"cls2":>6} {"cls3":>6} {"cls4":>6}  {"cls2_bar"}'
    print(hdr)
    print(f'  {"(water)":>56} {"(%)":>6} {"(%)":>6} {"★%":>6} {"(%)":>6} {"(%)":>6}')
    print('  ' + '─' * 97)
    for r in records:
        ratios = r['ratios']
        pcts = [ratios[c] * 100 for c in range(N_TRAIN_CLS)]
        b = bar(ratios[2], width=16)
        print(f'  {r["name"]:<55} '
              f'{pcts[0]:>5.1f}% {pcts[1]:>5.1f}% {pcts[2]:>5.1f}% '
              f'{pcts[3]:>5.1f}% {pcts[4]:>5.1f}%  {b}')


def aggregate_dist(records: list[dict]) -> np.ndarray:
    """按像素数加权聚合，得到整体类别分布向量。"""
    total = np.zeros(N_TRAIN_CLS, dtype=np.float64)
    for r in records:
        for c in range(N_TRAIN_CLS):
            total[c] += r['counts'][c]
    s = total.sum()
    return total / s if s > 0 else total


def print_dist_comparison(train_dist: np.ndarray, val_dist: np.ndarray,
                          val_label: str = 'Val'):
    print(f'\n{"─"*70}')
    print(f'  整体类别分布对比（像素加权）')
    print(f'{"─"*70}')
    print(f'  {"Class":<22} {"Train":>8}  {"bar(train)":<22} {"→":>2} '
          f'{val_label:>8}  {"bar(val)":<22}')
    print('  ' + '─' * 68)
    for c in range(N_TRAIN_CLS):
        tp, vp = train_dist[c] * 100, val_dist[c] * 100
        marker = '  ★' if c == 2 else ''
        print(f'  {TRAIN_CLS_NAMES[c]:<22} {tp:>7.2f}%  {bar(train_dist[c]):<22} '
              f'{"→":>2} {vp:>7.2f}%  {bar(val_dist[c]):<22}{marker}')

    kl = kl_divergence(train_dist, val_dist)
    js = js_divergence(train_dist, val_dist)
    print(f'\n  KL(train || {val_label}) = {kl:.4f}  '
          f'（< 0.05 好，0.05-0.15 一般，> 0.15 差异显著）')
    print(f'  JS散度            = {js:.4f}  '
          f'（0=完全相同，1=完全不同；< 0.05 为良好）')


def print_val_anomaly(train_records: list[dict], val_records: list[dict]):
    """找出 val 场景中 cls2/cls3 比例与训练集均值差异过大的场景。"""
    if not train_records or not val_records:
        return

    # 训练集按场景统计的均值和标准差
    train_cls2 = np.array([r['ratios'][2] for r in train_records])
    train_cls3 = np.array([r['ratios'][3] for r in train_records])
    mu2, sd2 = train_cls2.mean(), train_cls2.std() + 1e-9
    mu3, sd3 = train_cls3.mean(), train_cls3.std() + 1e-9

    print(f'\n{"─"*80}')
    print(f'  验证场景异常检测（cls2/cls3 相对训练集分布的 z-score）')
    print(f'  训练集 cls2: 均值={mu2*100:.1f}%  标准差={sd2*100:.1f}%')
    print(f'  训练集 cls3: 均值={mu3*100:.1f}%  标准差={sd3*100:.1f}%')
    print(f'{"─"*80}')
    print(f'  {"Scene":<55} {"cls2%":>6}  {"z2":>6}  {"cls3%":>6}  {"z3":>6}  {"状态"}')
    print('  ' + '─' * 78)

    for r in val_records:
        v2 = r['ratios'][2]
        v3 = r['ratios'][3]
        z2 = (v2 - mu2) / sd2
        z3 = (v3 - mu3) / sd3
        flags = []
        if v2 < 1e-4:
            flags.append('⚠ cls2 缺失')
        elif abs(z2) > 2:
            flags.append(f'⚠ cls2 偏离{z2:+.1f}σ')
        if v3 < 1e-4:
            flags.append('⚠ cls3 缺失')
        elif abs(z3) > 2:
            flags.append(f'⚠ cls3 偏离{z3:+.1f}σ')
        status = '  '.join(flags) if flags else '✓ 正常'
        print(f'  {r["name"]:<55} {v2*100:>5.1f}%  {z2:>+6.2f}  '
              f'{v3*100:>5.1f}%  {z3:>+6.2f}  {status}')


def print_season_dist(train_records: list[dict], val_records: list[dict]):
    def season_counts(records):
        cnt = defaultdict(int)
        for r in records:
            s = SEASON.get(r['month'], 'Unknown')
            cnt[s] += 1
        return cnt

    tc = season_counts(train_records)
    vc = season_counts(val_records)
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn', 'Unknown']
    n_train = len(train_records) or 1
    n_val = len(val_records) or 1

    print(f'\n{"─"*60}')
    print(f'  季节分布对比（按场景数量）')
    print(f'{"─"*60}')
    print(f'  {"Season":<10} {"Train":>8}  {"Val":>8}  {"差异"}')
    print('  ' + '─' * 58)
    for s in seasons:
        tp = tc[s] / n_train * 100
        vp = vc[s] / n_val * 100
        diff = vp - tp
        flag = '  ⚠ 过多' if diff > 20 else ('  ⚠ 过少' if diff < -20 else '')
        print(f'  {s:<10} {tc[s]:>5} ({tp:>4.0f}%)  '
              f'{vc[s]:>5} ({vp:>4.0f}%)  {diff:>+5.0f}%{flag}')


def save_csv(train_records: list[dict], val_records: list[dict], path: str):
    fieldnames = ['split', 'name', 'month', 'season', 'valid_px'] + \
                 [f'cls{c}_px' for c in range(N_TRAIN_CLS)] + \
                 [f'cls{c}_pct' for c in range(N_TRAIN_CLS)]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for split, records in (('train', train_records), ('val', val_records)):
            for r in records:
                row = {'split': split, 'name': r['name'],
                       'month': r['month'],
                       'season': SEASON.get(r['month'], 'Unknown'),
                       'valid_px': r['valid_px']}
                for c in range(N_TRAIN_CLS):
                    row[f'cls{c}_px'] = r['counts'][c]
                    row[f'cls{c}_pct'] = round(r['ratios'][c] * 100, 2)
                w.writerow(row)
    print(f'\n  完整数据已保存至: {path}')


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='验证集代表性检验：对比训练集与验证集的 SOD 类别分布。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data-dir', required=True,
                        help='包含 NC 数据文件的目录。')
    parser.add_argument('--train-list', required=True,
                        help='训练集 JSON 列表文件路径。')
    parser.add_argument('--val-list', nargs='+', required=True,
                        help='验证集 JSON 列表文件路径（可多个，逐一分析）。')
    parser.add_argument('--sod-var', default='SOD',
                        help='NC 文件中 SOD 变量名（默认：SOD）。')
    parser.add_argument('--mask-var', default='global_valid_mask',
                        help='有效像素掩膜变量名（默认：global_valid_mask）。')
    parser.add_argument('--output', default=None,
                        help='保存完整逐场景统计为 CSV 的路径（可选）。')
    args = parser.parse_args()

    # ── 读取场景列表 ─────────────────────────────────────────────────────────
    train_files_raw = load_json_list(args.train_list)
    # 合并所有 val 列表中的场景（去重）
    val_files_all: dict[str, str] = {}   # name → val_list_source
    for vl in args.val_list:
        for fn in load_json_list(vl):
            if fn not in val_files_all:
                val_files_all[fn] = os.path.basename(vl)

    # 从训练集中剔除出现在验证集中的场景
    val_set = set(val_files_all.keys())
    train_files = list(dict.fromkeys(   # 去重保序
        fn for fn in train_files_raw if fn not in val_set))

    print(f'\n训练集: {len(train_files)} 个场景（去重后，已排除 val 场景）')
    print(f'验证集: {len(val_files_all)} 个场景（来自 {len(args.val_list)} 个列表）')

    # ── 扫描训练集 ────────────────────────────────────────────────────────────
    print('\n正在扫描训练集...')
    train_records = []
    for fn in tqdm(train_files, desc='Train'):
        path = os.path.join(args.data_dir, fn)
        r = scene_class_dist(path, args.sod_var, args.mask_var)
        if r is not None:
            train_records.append(r)

    # ── 扫描验证集 ────────────────────────────────────────────────────────────
    print('\n正在扫描验证集...')
    val_records = []
    for fn in tqdm(list(val_files_all.keys()), desc='Val'):
        path = os.path.join(args.data_dir, fn)
        r = scene_class_dist(path, args.sod_var, args.mask_var)
        if r is not None:
            r['val_source'] = val_files_all[fn]
            val_records.append(r)

    if not train_records or not val_records:
        print('数据不足，退出。')
        return

    # ── 报告：逐场景表格 ──────────────────────────────────────────────────────
    print_scene_table(sorted(train_records, key=lambda x: -x['ratios'][2]),
                      '训练集（按 cls2 比例降序）')
    print_scene_table(sorted(val_records, key=lambda x: -x['ratios'][2]),
                      '验证集（按 cls2 比例降序）')

    # ── 报告：整体分布对比 ────────────────────────────────────────────────────
    train_dist = aggregate_dist(train_records)
    val_dist = aggregate_dist(val_records)
    print_dist_comparison(train_dist, val_dist, val_label='Val')

    # ── 如果提供了多个 val 列表，分别对比 ────────────────────────────────────
    if len(args.val_list) > 1:
        for vl in args.val_list:
            vl_name = os.path.basename(vl)
            sub = [r for r in val_records if r.get('val_source') == vl_name]
            if sub:
                vd = aggregate_dist(sub)
                print_dist_comparison(train_dist, vd, val_label=vl_name)

    # ── 报告：异常检测 ────────────────────────────────────────────────────────
    print_val_anomaly(train_records, val_records)

    # ── 报告：季节分布 ────────────────────────────────────────────────────────
    print_season_dist(train_records, val_records)

    # ── 报告：推荐权重 ────────────────────────────────────────────────────────
    print(f'\n{"─"*60}')
    print('  基于实际像素分布推荐的损失函数权重')
    print(f'{"─"*60}')
    # 按训练集整体分布，使用 1/sqrt(freq) 折中权重
    weights_raw = 1.0 / (np.sqrt(train_dist) + 1e-9)
    # 归一化到 cls0 = 1.0
    weights_norm = weights_raw / weights_raw[0]
    # 限制最大值，避免极端权重
    weights_clipped = np.clip(weights_norm, 0.5, 5.0)
    print(f'  1/sqrt(freq) 折中权重（归一化到 cls0=1）：')
    for c in range(N_TRAIN_CLS):
        bar_w = bar(weights_clipped[c] / 5.0, width=15)
        print(f'    cls{c}: {weights_clipped[c]:.3f}  {bar_w}  {TRAIN_CLS_NAMES[c]}')
    fmt = ', '.join(f'{w:.2f}' for w in weights_clipped)
    print(f'\n  → 建议配置：\'weight\': [{fmt}]')
    print(f'  注：这是基于训练集像素分布的参考值，需结合实际验证效果微调。')

    # ── 保存 CSV ──────────────────────────────────────────────────────────────
    if args.output:
        save_csv(train_records, val_records, args.output)


if __name__ == '__main__':
    main()
