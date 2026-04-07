#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""split_dataset.py
====================
扫描 NC 数据集，基于类别分布和季节进行分层抽样，生成平衡的训练集/验证集 JSON 列表。

分层策略（按稀缺程度从高到低）
--------------------------------
  cls2_cls3_mixed  : cls2 > 5% AND cls3 > 5%   → 强制进入训练集（最稀缺的混合场景）
  cls2_dominant    : cls2 > 30%                 → 绝大多数进训练集，至多1个进验证集
  cls3_dominant    : cls3 > 30%                 → 绝大多数进训练集，至多1个进验证集
  cls2_minor       : 1% < cls2 ≤ 30%（无大量cls3）→ 按比例分配，验证集保留代表
  cls4_heavy       : cls4 > 30%                 → 按正常比例分配
  cls1_heavy       : cls1 > 30%（cls2/cls3 < 1%）→ 按正常比例分配
  cls0_dominant    : cls0 > 80%                 → 可适当多分配到验证集
  other            : 其他混合场景               → 按正常比例分配

SOD 原始值 → 训练类别映射
--------------------------
  raw 0 → cls0  水体
  raw 1 → cls1  新冰（raw 1+2 合并）
  raw 2 → cls1
  raw 3 → cls2  薄一年冰  ★
  raw 4 → cls3  中厚一年冰
  raw 5 → cls4  多年冰

用法
----
  # 第一步：扫描指定 JSON 列表中的场景（结果保存为 CSV，可复用）
  python split_dataset.py scan \\
      --data-dir /root/autodl-tmp/My_dataset/ \\
      --file-list datalists/all_scenes.json \\
      --output scan_all.csv

  # 同时扫描多个 JSON 列表（自动去重）：
  python split_dataset.py scan \\
      --data-dir /root/autodl-tmp/My_dataset/ \\
      --file-list datalists/train_list_small.json datalists/val_list2.json \\
      --output scan_all.csv

  # 第二步：基于扫描结果生成划分方案
  python split_dataset.py split \\
      --scan-csv scan_all.csv \\
      --val-ratio 0.20 \\
      --output-train datalists/train_new.json \\
      --output-val datalists/val_new.json

  # 两步合并（直接扫描并划分，需提供 --file-list）：
  python split_dataset.py split \\
      --data-dir /root/autodl-tmp/My_dataset/ \\
      --file-list datalists/all_scenes.json \\
      --val-ratio 0.20 \\
      --output-train datalists/train_new.json \\
      --output-val datalists/val_new.json

  # 强制指定某些场景（用文件名关键词，逗号分隔）：
  python split_dataset.py split \\
      --scan-csv scan_all.csv \\
      --force-train 20250303,20250202,20210111 \\
      --force-val 20251229,20250318 \\
      --output-train datalists/train_new.json \\
      --output-val datalists/val_new.json
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
# 常量
# ---------------------------------------------------------------------------
RAW_TO_TRAIN = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4}
N_CLS = 5
CLS_NAMES = ['water', 'new/young', 'thin FYI(cls2)', 'med FYI(cls3)', 'MY ice(cls4)']
MASK_VAL = 255
SEASON_MAP = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn',
}


# ---------------------------------------------------------------------------
# NC 读取
# ---------------------------------------------------------------------------

def open_nc(path: str):
    for engine in ('h5netcdf', 'netcdf4', None):
        try:
            kw = {'engine': engine} if engine else {}
            return xr.open_dataset(path, mask_and_scale=False, **kw)
        except Exception:
            continue
    raise RuntimeError(f'Cannot open: {path}')


def scan_scene(nc_path: str, sod_var: str = 'SOD',
               mask_var: str = 'global_valid_mask') -> dict | None:
    """读取一个 NC 文件，返回类别像素统计（含场景总像素数和有效覆盖率）。"""
    try:
        ds = open_nc(nc_path)
    except Exception as e:
        print(f'  [WARN] {os.path.basename(nc_path)}: {e}', file=sys.stderr)
        return None

    if sod_var not in ds:
        ds.close()
        return None

    sod = ds[sod_var].values.flatten().astype(np.int32)
    total_px = len(sod)                      # 场景原始总像素数
    valid = sod != MASK_VAL
    if mask_var in ds:
        gvm = ds[mask_var].values.flatten().astype(np.float32)
        valid &= gvm > 0.5
    ds.close()

    counts = {c: 0 for c in range(N_CLS)}
    for raw, tcls in RAW_TO_TRAIN.items():
        counts[tcls] += int(((sod == raw) & valid).sum())

    valid_px = int(valid.sum())
    name = os.path.basename(nc_path)
    try:
        month = int(name[21:23])
    except (ValueError, IndexError):
        month = -1

    ratios = {c: counts[c] / valid_px if valid_px > 0 else 0.0
              for c in range(N_CLS)}
    valid_coverage = valid_px / total_px if total_px > 0 else 0.0

    return {
        'name': name,
        'total_px': total_px,
        'valid_px': valid_px,
        'valid_coverage': valid_coverage,
        'counts': counts,
        'ratios': ratios,
        'month': month,
        'season': SEASON_MAP.get(month, 'Unknown'),
    }


# ---------------------------------------------------------------------------
# 场景质量过滤
# ---------------------------------------------------------------------------

def qualify_scene(
    r: dict,
    min_valid_coverage: float = 0.20,
    max_water_ratio: float = 0.98,
    min_class_pixels: int = 500,
) -> tuple[bool, str]:
    """
    判断场景是否达标。返回 (is_qualified, discard_reason)。

    过滤条件：
      1. valid_coverage >= min_valid_coverage  : 有效像素覆盖率不能太低
      2. cls0_ratio <= max_water_ratio         : 不能几乎全是水体
      3. 每个存在的非水类别像素数 >= min_class_pixels : 排除标注噪声
    """
    cov = r.get('valid_coverage', 1.0)
    if cov < min_valid_coverage:
        return False, (f'有效覆盖率 {cov*100:.1f}% '
                       f'< 阈值 {min_valid_coverage*100:.0f}%')

    water_ratio = r['ratios'][0]
    if water_ratio > max_water_ratio:
        return False, (f'水体占比 {water_ratio*100:.1f}% '
                       f'> 阈值 {max_water_ratio*100:.0f}%，场景几乎全为水体')

    for c in range(1, N_CLS):
        px = r['counts'][c]
        if 0 < px < min_class_pixels:
            return False, (f'cls{c}({CLS_NAMES[c]}) 仅有 {px} 像素 '
                           f'< 阈值 {min_class_pixels}，疑似标注噪声')

    return True, ''


# ---------------------------------------------------------------------------
# 场景分层
# ---------------------------------------------------------------------------

def classify_stratum(r: dict) -> str:
    """根据类别比例为场景分配分层标签。"""
    c2 = r['ratios'][2]
    c3 = r['ratios'][3]
    c4 = r['ratios'][4]
    c1 = r['ratios'][1]
    c0 = r['ratios'][0]

    if c2 > 0.05 and c3 > 0.05:
        return 'cls2_cls3_mixed'   # 最稀缺，强制训练集
    if c2 > 0.30:
        return 'cls2_dominant'
    if c3 > 0.30:
        return 'cls3_dominant'
    if 0.01 < c2 <= 0.30:
        return 'cls2_minor'
    if c4 > 0.30:
        return 'cls4_heavy'
    if c1 > 0.30:
        return 'cls1_heavy'
    if c0 > 0.80:
        return 'cls0_dominant'
    return 'other'


# 每个分层的验证集分配策略
# 'max_val_count': 最多放多少个到验证集（None=无限制，按比例来）
# 'min_val_count': 验证集至少保留几个（仅在数量充足时）
# 'force_train':   True=强制所有场景进训练集
STRATUM_POLICY = {
    'cls2_cls3_mixed': {'force_train': True,  'max_val_count': 0},
    'cls2_dominant':   {'force_train': False, 'max_val_count': 1},
    'cls3_dominant':   {'force_train': False, 'max_val_count': 1},
    'cls2_minor':      {'force_train': False, 'max_val_count': None},
    'cls4_heavy':      {'force_train': False, 'max_val_count': None},
    'cls1_heavy':      {'force_train': False, 'max_val_count': None},
    'cls0_dominant':   {'force_train': False, 'max_val_count': None},
    'other':           {'force_train': False, 'max_val_count': None},
}

# 验证集期望拥有的最低 cls2 场景数
VAL_MIN_CLS2_SCENES = 2
# 验证集期望每个季节至少有几个场景
VAL_TARGET_PER_SEASON = 1


# ---------------------------------------------------------------------------
# 划分算法
# ---------------------------------------------------------------------------

def propose_split(
    records: list[dict],
    val_ratio: float = 0.20,
    force_train_keys: list[str] = None,
    force_val_keys: list[str] = None,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    分层抽样，返回 (train_records, val_records)。

    force_train_keys / force_val_keys: 包含这些关键词的场景名强制归入对应集合。
    """
    rng = np.random.default_rng(seed)
    force_train_keys = force_train_keys or []
    force_val_keys = force_val_keys or []

    # ── 1. 强制指定场景 ───────────────────────────────────────────────────────
    forced_train, forced_val, free = [], [], []
    for r in records:
        if any(k in r['name'] for k in force_val_keys):
            forced_val.append(r)
        elif any(k in r['name'] for k in force_train_keys):
            forced_train.append(r)
        else:
            free.append(r)

    # ── 2. 分层 ───────────────────────────────────────────────────────────────
    strata: dict[str, list[dict]] = defaultdict(list)
    for r in free:
        r['stratum'] = classify_stratum(r)
        strata[r['stratum']].append(r)

    train_pool, val_pool = list(forced_train), list(forced_val)
    for r in forced_train:
        r['stratum'] = classify_stratum(r)
    for r in forced_val:
        r['stratum'] = classify_stratum(r)

    # ── 3. 按分层策略分配 ────────────────────────────────────────────────────
    for stratum, scenes in strata.items():
        policy = STRATUM_POLICY[stratum]

        if policy['force_train']:
            train_pool.extend(scenes)
            continue

        # 在分层内按季节打乱，保证季节多样性
        scenes_sorted = sorted(scenes, key=lambda x: (x['season'], rng.random()))
        n_total = len(scenes_sorted)
        n_val_target = round(n_total * val_ratio)

        max_v = policy['max_val_count']
        if max_v is not None:
            n_val_target = min(n_val_target, max_v)

        # 尽量让 val 中各季节均有代表
        val_scenes = _season_balanced_pick(scenes_sorted, n_val_target, rng)
        train_scenes = [s for s in scenes_sorted if s not in val_scenes]

        train_pool.extend(train_scenes)
        val_pool.extend(val_scenes)

    # ── 4. 验证集 cls2 覆盖补救 ───────────────────────────────────────────────
    # 如果验证集没有足够的 cls2 场景，从训练集中补充
    val_cls2 = [r for r in val_pool if r['ratios'][2] > 0.01]
    if len(val_cls2) < VAL_MIN_CLS2_SCENES:
        # 从训练集中找 cls2 > 1% 且不是 cls2_cls3_mixed 的场景
        candidates = [r for r in train_pool
                      if r['ratios'][2] > 0.01
                      and r.get('stratum') != 'cls2_cls3_mixed']
        # 按 cls2% 降序，选最具代表性的
        candidates.sort(key=lambda x: -x['ratios'][2])
        need = VAL_MIN_CLS2_SCENES - len(val_cls2)
        to_move = candidates[:need]
        for r in to_move:
            train_pool.remove(r)
            val_pool.append(r)

    # ── 5. 验证集季节补救 ────────────────────────────────────────────────────
    val_seasons = {r['season'] for r in val_pool}
    train_seasons_available = defaultdict(list)
    for r in train_pool:
        train_seasons_available[r['season']].append(r)

    for season in ('Winter', 'Spring', 'Summer', 'Autumn'):
        if season not in val_seasons and train_seasons_available[season]:
            # 从训练集中把这个季节的一个 cls0_dominant 场景移到验证集
            candidates = [r for r in train_seasons_available[season]
                          if r.get('stratum') == 'cls0_dominant']
            if not candidates:
                candidates = train_seasons_available[season]
            r = rng.choice(candidates)
            train_pool.remove(r)
            val_pool.append(r)

    return train_pool, val_pool


def _season_balanced_pick(scenes: list[dict], n: int, rng) -> list[dict]:
    """从 scenes 中选 n 个，尽量每个季节都有代表。"""
    if n == 0:
        return []
    if n >= len(scenes):
        return list(scenes)

    by_season: dict[str, list] = defaultdict(list)
    for s in scenes:
        by_season[s['season']].append(s)

    selected = []
    seasons = list(by_season.keys())
    rng.shuffle(seasons)

    # 轮询每个季节各取一个
    for season in seasons:
        if len(selected) >= n:
            break
        pool = by_season[season]
        if pool:
            chosen = rng.choice(pool)
            selected.append(chosen)

    # 如果还不够，从剩余场景随机补充
    remaining = [s for s in scenes if s not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[:n - len(selected)])

    return selected[:n]


# ---------------------------------------------------------------------------
# CSV 读写
# ---------------------------------------------------------------------------

CSV_FIELDS = ['name', 'total_px', 'valid_px', 'valid_coverage_pct', 'month', 'season',
              'cls0_px', 'cls1_px', 'cls2_px', 'cls3_px', 'cls4_px',
              'cls0_pct', 'cls1_pct', 'cls2_pct', 'cls3_pct', 'cls4_pct']


def save_scan_csv(records: list[dict], path: str):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in records:
            row = {
                'name': r['name'],
                'total_px': r.get('total_px', ''),
                'valid_px': r['valid_px'],
                'valid_coverage_pct': round(r.get('valid_coverage', 1.0) * 100, 2),
                'month': r['month'],
                'season': r['season'],
            }
            for c in range(N_CLS):
                row[f'cls{c}_px'] = r['counts'][c]
                row[f'cls{c}_pct'] = round(r['ratios'][c] * 100, 3)
            w.writerow(row)
    print(f'扫描结果已保存至: {path}')


def load_scan_csv(path: str) -> list[dict]:
    records = []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            valid_px = int(row['valid_px'])
            counts = {c: int(row[f'cls{c}_px']) for c in range(N_CLS)}
            ratios = {c: counts[c] / valid_px if valid_px > 0 else 0.0
                      for c in range(N_CLS)}
            total_px = int(row['total_px']) if row.get('total_px') else valid_px
            cov_pct = float(row['valid_coverage_pct']) if row.get('valid_coverage_pct') else 100.0
            records.append({
                'name': row['name'],
                'total_px': total_px,
                'valid_px': valid_px,
                'valid_coverage': cov_pct / 100.0,
                'month': int(row['month']),
                'season': row['season'],
                'counts': counts,
                'ratios': ratios,
            })
    return records


# ---------------------------------------------------------------------------
# 报告打印
# ---------------------------------------------------------------------------

def bar(ratio: float, width: int = 16) -> str:
    n = int(ratio * width)
    return '█' * n + '░' * (width - n)


def print_split_report(train: list[dict], val: list[dict]):
    def agg(records):
        total = np.zeros(N_CLS)
        for r in records:
            for c in range(N_CLS):
                total[c] += r['counts'][c]
        s = total.sum()
        return total / s if s > 0 else total

    def season_cnt(records):
        cnt = defaultdict(int)
        for r in records:
            cnt[r['season']] += 1
        return cnt

    def stratum_cnt(records):
        cnt = defaultdict(int)
        for r in records:
            cnt[r.get('stratum', classify_stratum(r))] += 1
        return cnt

    print(f'\n{"═"*100}')
    print(f'  划分结果汇总')
    print(f'{"═"*100}')
    print(f'  训练集: {len(train)} 个场景    验证集: {len(val)} 个场景    '
          f'验证比例: {len(val)/(len(train)+len(val))*100:.1f}%')

    # 类别分布对比
    tr_dist = agg(train)
    va_dist = agg(val)
    print(f'\n  {"类别":<22} {"训练集":>8}  {"bar(train)":<18}  {"验证集":>8}  {"bar(val)":<18}')
    print('  ' + '─' * 78)
    for c in range(N_CLS):
        mark = '  ★' if c == 2 else ''
        print(f'  cls{c} {CLS_NAMES[c]:<17} {tr_dist[c]*100:>7.2f}%  {bar(tr_dist[c]):<18}  '
              f'{va_dist[c]*100:>7.2f}%  {bar(va_dist[c]):<18}{mark}')

    # 季节分布
    tc = season_cnt(train)
    vc = season_cnt(val)
    print(f'\n  {"季节":<10} {"训练集":>8}  {"验证集":>8}')
    print('  ' + '─' * 30)
    for s in ('Winter', 'Spring', 'Summer', 'Autumn'):
        flag = '  ⚠' if vc[s] == 0 else ''
        print(f'  {s:<10} {tc[s]:>8}  {vc[s]:>8}{flag}')

    # 分层统计
    tc2 = stratum_cnt(train)
    vc2 = stratum_cnt(val)
    print(f'\n  {"分层":<20} {"训练集":>8}  {"验证集":>8}')
    print('  ' + '─' * 40)
    for st in STRATUM_POLICY:
        force_mark = '  [强制训练]' if STRATUM_POLICY[st]['force_train'] else ''
        print(f'  {st:<20} {tc2[st]:>8}  {vc2[st]:>8}{force_mark}')

    # 验证集逐场景
    print(f'\n  验证集场景列表（按 cls2% 降序）:')
    print(f'  {"Scene":<56} {"cls2%":>6}  {"cls3%":>6}  {"季节":<8}  {"分层"}')
    print('  ' + '─' * 96)
    for r in sorted(val, key=lambda x: -x['ratios'][2]):
        st = r.get('stratum', classify_stratum(r))
        print(f'  {r["name"]:<56} {r["ratios"][2]*100:>5.1f}%  '
              f'{r["ratios"][3]*100:>5.1f}%  {r["season"]:<8}  {st}')

    # 训练集 cls2 场景
    train_cls2 = [r for r in train if r['ratios'][2] > 0.01]
    print(f'\n  训练集中含 cls2 (>1%) 的场景:')
    if train_cls2:
        print(f'  {"Scene":<56} {"cls2%":>6}  {"cls3%":>6}  {"季节":<8}  {"分层"}')
        print('  ' + '─' * 96)
        for r in sorted(train_cls2, key=lambda x: -x['ratios'][2]):
            st = r.get('stratum', classify_stratum(r))
            print(f'  {r["name"]:<56} {r["ratios"][2]*100:>5.1f}%  '
                  f'{r["ratios"][3]*100:>5.1f}%  {r["season"]:<8}  {st}')
    else:
        print('  ⚠  训练集中没有含 cls2 的场景！请检查数据或使用 --force-train 调整。')

    print(f'\n{"═"*100}')


def print_discard_report(discarded: list[tuple[dict, str]]):
    """打印被过滤丢弃的场景列表。"""
    if not discarded:
        print('\n  [过滤] 无场景被丢弃，全部达标。')
        return
    print(f'\n{"─"*90}')
    print(f'  被丢弃场景（共 {len(discarded)} 个）:')
    print(f'  {"Scene":<56} {"丢弃原因"}')
    print('  ' + '─' * 88)
    for r, reason in discarded:
        print(f'  {r["name"]:<56} {reason}')


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def cmd_scan(args):
    """扫描所有 NC 文件并保存统计 CSV。"""
    filenames = []
    seen = set()
    for jf in args.file_list:
        with open(jf, encoding='utf-8') as f:
            for fn in json.load(f):
                if fn not in seen:
                    seen.add(fn)
                    filenames.append(fn)
    files = [os.path.join(args.data_dir, fn) for fn in filenames]
    print(f'从 JSON 列表加载 {len(files)} 个场景（去重后）...')

    records = []
    for path in tqdm(files, desc='扫描 NC 文件'):
        r = scan_scene(path, sod_var=args.sod_var, mask_var=args.mask_var or '')
        if r is not None:
            records.append(r)

    print(f'\n成功扫描 {len(records)} 个场景，开始质量过滤...')

    qualified, discarded = [], []
    for r in records:
        ok, reason = qualify_scene(
            r,
            min_valid_coverage=args.min_valid_coverage,
            max_water_ratio=args.max_water_ratio,
            min_class_pixels=args.min_class_pixels,
        )
        if ok:
            qualified.append(r)
        else:
            discarded.append((r, reason))

    print_discard_report(discarded)
    print(f'\n  达标场景: {len(qualified)} 个  丢弃: {len(discarded)} 个')

    # 打印简要统计
    cls2_scenes = [r for r in qualified if r['ratios'][2] > 0.01]
    print(f'  含 cls2 (>1%) 场景数: {len(cls2_scenes)}')
    for st_name in ('cls2_cls3_mixed', 'cls2_dominant', 'cls3_dominant'):
        cnt = sum(1 for r in qualified if classify_stratum(r) == st_name)
        print(f'  分层 {st_name}: {cnt} 个')

    save_scan_csv(qualified, args.output)


def cmd_split(args):
    """基于扫描结果或直接扫描，生成训练/验证集划分。"""
    # 获取 records
    if args.scan_csv and os.path.exists(args.scan_csv):
        print(f'加载扫描结果: {args.scan_csv}')
        records = load_scan_csv(args.scan_csv)
    elif args.data_dir:
        if not args.file_list:
            print('错误：未提供 --scan-csv 时，必须通过 --file-list 指定场景 JSON 列表。',
                  file=sys.stderr)
            sys.exit(1)
        filenames = []
        seen = set()
        for jf in args.file_list:
            with open(jf, encoding='utf-8') as f:
                for fn in json.load(f):
                    if fn not in seen:
                        seen.add(fn)
                        filenames.append(fn)
        files = [os.path.join(args.data_dir, fn) for fn in filenames]
        print(f'从 JSON 列表加载 {len(files)} 个场景（去重后），开始扫描...')
        records = []
        for path in tqdm(files, desc='扫描 NC 文件'):
            r = scan_scene(path, sod_var=args.sod_var, mask_var=args.mask_var or '')
            if r is not None:
                records.append(r)
    else:
        print('错误：请提供 --scan-csv 或 --data-dir。', file=sys.stderr)
        sys.exit(1)

    # 质量过滤
    qualified, discarded = [], []
    for r in records:
        ok, reason = qualify_scene(
            r,
            min_valid_coverage=args.min_valid_coverage,
            max_water_ratio=args.max_water_ratio,
            min_class_pixels=args.min_class_pixels,
        )
        if ok:
            qualified.append(r)
        else:
            discarded.append((r, reason))

    print_discard_report(discarded)
    print(f'\n达标场景: {len(qualified)} 个  丢弃: {len(discarded)} 个  '
          f'目标验证比例: {args.val_ratio:.0%}')
    records = qualified

    # 解析 force 列表
    force_train = [k.strip() for k in args.force_train.split(',') if k.strip()] \
        if args.force_train else []
    force_val = [k.strip() for k in args.force_val.split(',') if k.strip()] \
        if args.force_val else []

    if force_train:
        print(f'强制进训练集关键词: {force_train}')
    if force_val:
        print(f'强制进验证集关键词: {force_val}')

    # 执行划分
    train, val = propose_split(
        records,
        val_ratio=args.val_ratio,
        force_train_keys=force_train,
        force_val_keys=force_val,
        seed=args.seed,
    )

    # 打印报告
    print_split_report(train, val)

    # 保存 JSON
    if args.output_train:
        with open(args.output_train, 'w', encoding='utf-8') as f:
            json.dump([r['name'] for r in train], f, indent=4, ensure_ascii=False)
        print(f'训练集列表已保存至: {args.output_train}  ({len(train)} 个场景)')

    if args.output_val:
        with open(args.output_val, 'w', encoding='utf-8') as f:
            json.dump([r['name'] for r in val], f, indent=4, ensure_ascii=False)
        print(f'验证集列表已保存至: {args.output_val}  ({len(val)} 个场景)')

    if not args.output_train and not args.output_val:
        print('\n提示：使用 --output-train 和 --output-val 保存划分结果。')


def _add_filter_args(p):
    """向子命令 parser 添加场景质量过滤参数（scan/split 共用）。"""
    g = p.add_argument_group('场景质量过滤')
    g.add_argument('--min-valid-coverage', type=float, default=0.20,
                   metavar='RATIO',
                   help='有效像素覆盖率下限（默认 0.20）。'
                        '低于此值的场景被丢弃。设为 0 可关闭此过滤。')
    g.add_argument('--max-water-ratio', type=float, default=0.98,
                   metavar='RATIO',
                   help='水体(cls0)占比上限（默认 0.98）。'
                        '超过此值视为"几乎纯水"场景被丢弃。设为 1.0 可关闭。')
    g.add_argument('--min-class-pixels', type=int, default=500,
                   metavar='N',
                   help='非水类别最小像素数（默认 500）。'
                        '某类存在但像素数不足时视为标注噪声被丢弃。设为 0 可关闭。')


def build_parser():
    parser = argparse.ArgumentParser(
        description='数据集训练/验证集分层划分工具。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    # ── scan 子命令 ────────────────────────────────────────────────────────────
    p_scan = sub.add_parser('scan', help='扫描 NC 文件，统计类别分布，保存 CSV。')
    p_scan.add_argument('--data-dir', required=True,
                        help='NC 数据文件目录。')
    p_scan.add_argument('--file-list', nargs='+', required=True, metavar='JSON',
                        help='要扫描的场景 JSON 列表文件路径（可多个，自动去重）。'
                             '例如: --file-list datalists/all_scenes.json')
    p_scan.add_argument('--output', required=True,
                        help='输出 CSV 文件路径，例如 scan_all.csv。')
    p_scan.add_argument('--sod-var', default='SOD')
    p_scan.add_argument('--mask-var', default='global_valid_mask')
    _add_filter_args(p_scan)

    # ── split 子命令 ──────────────────────────────────────────────────────────
    p_spl = sub.add_parser('split', help='基于扫描结果生成训练/验证集划分。')
    p_spl.add_argument('--scan-csv', default=None,
                       help='scan 子命令生成的 CSV 文件路径。'
                            '若不提供则需要 --data-dir 重新扫描。')
    p_spl.add_argument('--data-dir', default=None,
                       help='NC 数据文件目录（--scan-csv 不存在时使用）。')
    p_spl.add_argument('--file-list', nargs='+', default=None, metavar='JSON',
                       help='要划分的场景 JSON 列表（可多个，自动去重）。'
                            '未提供 --scan-csv 时必填。')
    p_spl.add_argument('--val-ratio', type=float, default=0.20,
                       help='验证集比例，默认 0.20（20%%）。')
    p_spl.add_argument('--force-train', default=None,
                       help='强制进入训练集的场景名关键词，逗号分隔。'
                            '例如: --force-train 20250303,20250202')
    p_spl.add_argument('--force-val', default=None,
                       help='强制进入验证集的场景名关键词，逗号分隔。')
    p_spl.add_argument('--output-train', default=None,
                       help='训练集 JSON 输出路径。')
    p_spl.add_argument('--output-val', default=None,
                       help='验证集 JSON 输出路径。')
    p_spl.add_argument('--seed', type=int, default=42,
                       help='随机种子，保证可复现（默认 42）。')
    p_spl.add_argument('--sod-var', default='SOD')
    p_spl.add_argument('--mask-var', default='global_valid_mask')
    _add_filter_args(p_spl)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == 'scan':
        cmd_scan(args)
    elif args.cmd == 'split':
        cmd_split(args)


if __name__ == '__main__':
    main()
