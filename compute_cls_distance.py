#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""compute_cls_distance.py
===========================
计算训练集中 cls1（新冰/幼冰）、cls2（薄一年冰）、cls3（厚一年冰）
在 SAR 特征空间中的欧式距离，用于评估 cls2 标签的污染程度。

NC 文件 raw SOD 与训练类别对应关系（同 loaders.py 的 remap 逻辑）：
  raw 0       → cls 0  水体
  raw 1,2     → cls 1  新冰/幼冰（合并）
  raw 3       → cls 2  薄一年冰  ★
  raw 4       → cls 3  厚一年冰
  raw 5       → cls 4  多年冰
  raw 255     → 无效（mask）

输出：
  1. 各类别在特征空间的均值/标准差
  2. 类别质心之间的欧式距离（原始 & 归一化）
  3. cls2 像素污染分析：有多少 cls2 像素在特征空间中比 cls2 质心
     更接近 cls1 或 cls3 质心

Usage
-----
    python compute_cls_distance.py \\
        --data-dir F:/ZJU/11_Ice/dataset_create/dataset_nc_new \\
        --file-list datalists/train_list_small.json \\
        --features HH HV \\
        --max-px-per-scene 5000 \\
        --output cls_distance_report.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import xarray as xr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 特征变量名（对应 NC 文件中的变量）
# ---------------------------------------------------------------------------
FEATURE_META = {
    'HH':          'nersc_sar_primary',
    'HV':          'nersc_sar_secondary',
    'IA':          'sar_incidenceangle',
    'GLCM_CON':    'glcm_sigma0_hh_contrast',
    'GLCM_DIS':    'glcm_sigma0_hh_dissimilarity',
    'GLCM_HOM':    'glcm_sigma0_hh_homogeneity',
}

# NC raw SOD → training class
# raw 1,2 → cls1 ; raw 3 → cls2 ; raw 4 → cls3
RAW_TO_TRAIN = {1: 1, 2: 1, 3: 2, 4: 3}
TARGET_CLASSES = [1, 2, 3]   # cls1, cls2, cls3


def open_nc(path: str):
    for engine in ['h5netcdf', 'netcdf4', None]:
        try:
            kw = {'engine': engine} if engine else {}
            return xr.open_dataset(path, mask_and_scale=False, **kw)
        except Exception:
            continue
    raise RuntimeError(f"Cannot open: {path}")


def resolve_nc_path(scene_name: str, data_dirs: list[str]) -> str | None:
    """在多个数据目录中查找场景 NC 文件，返回首个存在的路径。"""
    if os.path.isabs(scene_name):
        return scene_name if os.path.exists(scene_name) else None

    for d in data_dirs:
        candidate = os.path.join(d, scene_name)
        if os.path.exists(candidate):
            return candidate
    return None


def extract_pixels(
    nc_path: str,
    feature_vars: list[str],
    max_px: int,
    valid_mask_var: str = 'global_valid_mask',
    append_ia: bool = False,
) -> dict[int, np.ndarray] | None:
    """
    从一个 NC 文件中提取三类像素的特征向量。

    Parameters
    ----------
    append_ia : bool
        若为 True，在特征列最后追加入射角（sar_incidenceangle）列，
        用于后续 IA 校正。追加的 IA 列不算在 feature_vars 里，
        调用方需自行记录其索引位置（即 -1 列）。

    Returns
    -------
    dict: {train_cls: ndarray(N, F[+1])} 或 None（文件读取失败）
    """
    IA_VAR = FEATURE_META['IA']   # 'sar_incidenceangle'

    try:
        ds = open_nc(nc_path)
    except Exception as e:
        print(f"  Skip {os.path.basename(nc_path)}: {e}", file=sys.stderr)
        return None

    # 检查所需变量是否存在
    vars_needed = feature_vars + ['SOD']
    if append_ia and IA_VAR not in feature_vars:
        vars_needed = vars_needed + [IA_VAR]
    missing = [v for v in vars_needed if v not in ds]
    if missing:
        print(f"  Skip {os.path.basename(nc_path)}: missing vars {missing}", file=sys.stderr)
        ds.close()
        return None

    # 读取 SOD 标签
    sod_raw = ds['SOD'].values.flatten().astype(np.int32)   # (H*W,)

    # 构建全局有效像素掩码
    valid_base = sod_raw != 255
    if valid_mask_var in ds:
        gvm = ds[valid_mask_var].values.flatten()
        valid_base = valid_base & (gvm > 0.5)

    # 读取特征，按需追加 IA
    vars_to_read = list(feature_vars)
    if append_ia and IA_VAR not in feature_vars:
        vars_to_read = vars_to_read + [IA_VAR]

    feat_array = np.stack(
        [ds[v].values.flatten().astype(np.float32) for v in vars_to_read],
        axis=0
    )  # (F[+1], N_total)
    ds.close()

    # 按 training class 分组提取
    result: dict[int, np.ndarray] = {}
    for raw_val, train_cls in RAW_TO_TRAIN.items():
        if train_cls not in TARGET_CLASSES:
            continue
        mask = valid_base & (sod_raw == raw_val)
        if train_cls not in result:
            result[train_cls] = feat_array[:, mask].T      # (N, F[+1])
        else:
            result[train_cls] = np.concatenate(
                [result[train_cls], feat_array[:, mask].T], axis=0)

    # 对每类随机下采样，防止单场景主导结果
    if max_px > 0:
        for cls in list(result.keys()):
            arr = result[cls]
            if len(arr) > max_px:
                idx = np.random.choice(len(arr), max_px, replace=False)
                result[cls] = arr[idx]

    return result


def ia_normalize(
    pixel_arrays: dict[int, np.ndarray],
    feature_names: list[str],
    ia_col_idx: int,
    ia_ref: float | None = None,
) -> tuple[dict[int, np.ndarray], dict, float]:
    """
    对受入射角影响的特征（HH、HV）做线性回归校正，
    移除入射角引起的系统性偏差。

    校正公式：
        feature_corr = feature - slope * (IA - IA_ref)

    其中 slope 由全体像素的 OLS 回归估计，IA_ref 为全体像素入射角均值。

    Parameters
    ----------
    pixel_arrays : {cls: ndarray(N, F+1)}
        最后一列为入射角（ia_col_idx 指定）。
    feature_names : list[str]
        特征名列表（不含 IA），用于判断哪些列需要校正。
    ia_col_idx : int
        IA 列在 pixel_arrays 中的列索引（通常为 -1）。
    ia_ref : float or None
        参考入射角（度）；None 时取全体像素 IA 均值。

    Returns
    -------
    corrected : {cls: ndarray(N, F)}  ← 已移除 IA 列
    slopes    : {feature_name: slope}  ← 各特征的回归斜率 (单位/度)
    ia_ref    : float                  ← 使用的参考入射角
    """
    IA_AFFECTED = {'HH', 'HV'}   # 仅校正这两个特征

    # 合并所有类的像素，用于估计全局斜率
    all_arr = np.concatenate(list(pixel_arrays.values()), axis=0)   # (N_all, F+1)
    ia_all  = all_arr[:, ia_col_idx]                                 # (N_all,)

    if ia_ref is None:
        ia_ref = float(ia_all.mean())

    # 构造特征列索引（非 IA 列）
    n_cols = all_arr.shape[1]
    feat_indices = [i for i in range(n_cols) if i != ia_col_idx % n_cols]

    # OLS：slope = cov(feat, IA) / var(IA)
    ia_centered = ia_all - ia_all.mean()
    ia_var = float(np.var(ia_centered)) + 1e-12
    slopes: dict[str, float] = {}

    for feat_idx, fname in zip(feat_indices, feature_names):
        if fname in IA_AFFECTED:
            feat_col = all_arr[:, feat_idx]
            slope = float(np.dot(ia_centered, feat_col - feat_col.mean()) / (len(ia_centered) * ia_var))
            slopes[fname] = slope
        else:
            slopes[fname] = 0.0   # 不校正

    # 对每类像素做校正，并移除 IA 列
    corrected: dict[int, np.ndarray] = {}
    for cls, arr in pixel_arrays.items():
        arr_c = arr[:, feat_indices].copy()    # 去掉 IA 列，(N, F)
        ia_col = arr[:, ia_col_idx]            # (N,)
        for feat_idx, fname in enumerate(feature_names):
            if slopes[fname] != 0.0:
                arr_c[:, feat_idx] -= slopes[fname] * (ia_col - ia_ref)
        corrected[cls] = arr_c

    return corrected, slopes, ia_ref


def compute_distance(a: np.ndarray, b: np.ndarray) -> float:
    """两个质心向量之间的欧式距离。"""
    return float(np.linalg.norm(a - b))


def analyze_cls2_pollution(
    cls2_pixels: np.ndarray,
    centroids: dict[int, np.ndarray],
) -> dict:
    """
    对每个 cls2 像素计算到三个质心的距离，
    统计"比 cls2 质心更接近 cls1 或 cls3"的比例。
    """
    c1, c2, c3 = centroids[1], centroids[2], centroids[3]

    d_to_c1 = np.linalg.norm(cls2_pixels - c1, axis=1)  # (N,)
    d_to_c2 = np.linalg.norm(cls2_pixels - c2, axis=1)
    d_to_c3 = np.linalg.norm(cls2_pixels - c3, axis=1)

    closer_to_c1 = (d_to_c1 < d_to_c2) & (d_to_c1 < d_to_c3)
    closer_to_c3 = (d_to_c3 < d_to_c2) & (d_to_c3 < d_to_c1)
    closer_to_either = closer_to_c1 | closer_to_c3

    n = len(cls2_pixels)
    return {
        'n_cls2':              n,
        'n_closer_to_cls1':    int(closer_to_c1.sum()),
        'n_closer_to_cls3':    int(closer_to_c3.sum()),
        'n_closer_to_either':  int(closer_to_either.sum()),
        'frac_closer_to_cls1': float(closer_to_c1.sum()) / n,
        'frac_closer_to_cls3': float(closer_to_c3.sum()) / n,
        'frac_polluted':       float(closer_to_either.sum()) / n,
        'mean_d_to_cls1':      float(d_to_c1.mean()),
        'mean_d_to_cls2':      float(d_to_c2.mean()),
        'mean_d_to_cls3':      float(d_to_c3.mean()),
    }


def print_report(
    feature_names: list[str],
    centroids: dict[int, np.ndarray],
    stds: dict[int, np.ndarray],
    counts: dict[int, int],
    dist_raw: dict[tuple, float],
    dist_norm: dict[tuple, float],
    pollution: dict,
    ia_slopes: dict[str, float] | None = None,
    ia_ref: float | None = None,
):
    cls_names = {1: 'cls1 (new/young ice)', 2: 'cls2 (thin FYI) ★', 3: 'cls3 (thick FYI)'}

    print("\n" + "="*70)
    print(" SAR 特征空间类别分析报告")
    print("="*70)

    # 0. 入射角校正信息
    if ia_slopes is not None:
        # 注意：若 NC 文件中 IA 已经过 z-score 归一化，ia_ref 为归一化后的均值（接近0），
        # slope 单位为"归一化特征值 / 归一化 IA 单位"，非物理单位 /°。
        ia_is_normalized = abs(ia_ref) < 5.0   # 真实入射角不会接近 0
        ia_unit_note = "（IA 已 z-score，slope 为归一化单位/归一化IA单位）" if ia_is_normalized \
                       else f"（跨 29° 幅宽校正幅度 ≈ {max(abs(s) for s in ia_slopes.values() if s!=0)*29:.3f}）"
        print(f"\n 入射角校正（IA_ref = {ia_ref:.4f}{'  ⚠ IA已归一化，非真实度数' if ia_is_normalized else '°'}）")
        print("─"*70)
        if ia_is_normalized:
            print(f"  ⚠  NC 文件中入射角已 z-score 归一化（均值≈0），slope 为归一化空间斜率，")
            print(f"     数学校正有效，但斜率数值不对应真实物理单位（°）。")
        for fname, slope in ia_slopes.items():
            if slope != 0.0:
                print(f"  {fname}: slope = {slope:.5f}")
            else:
                print(f"  {fname}: 不校正")
    else:
        print("\n ⚠  未启用入射角校正（--ia-correction 为 False）")

    # 1. 各类别特征统计
    print(f"\n{'特征':>18}", end="")
    for cls in TARGET_CLASSES:
        print(f"  {cls_names[cls]:>26}", end="")
    print()
    print("-"*70)

    for i, fname in enumerate(feature_names):
        print(f"{fname:>18}", end="")
        for cls in TARGET_CLASSES:
            mu = centroids[cls][i]
            sd = stds[cls][i]
            print(f"  {mu:>10.4f} ± {sd:>8.4f}", end="")
        print()

    print(f"\n{'像素数':>18}", end="")
    for cls in TARGET_CLASSES:
        print(f"  {counts[cls]:>26,}", end="")
    print()

    # 2. 类别间欧式距离
    print("\n" + "─"*70)
    print(" 类别质心欧式距离（原始特征空间）")
    print("─"*70)
    for (ci, cj), d in dist_raw.items():
        print(f"  dist({cls_names[ci]}, {cls_names[cj]}) = {d:.6f}")

    print("\n 类别质心欧式距离（z-score 归一化后）")
    print("─"*70)
    for (ci, cj), d in dist_norm.items():
        print(f"  dist({cls_names[ci]}, {cls_names[cj]}) = {d:.6f}")

    # 3. 是否两侧距离相近
    d12 = dist_norm[(1, 2)]
    d23 = dist_norm[(2, 3)]
    ratio = min(d12, d23) / max(d12, d23) if max(d12, d23) > 0 else 0
    print(f"\n  cls2 两侧距离比（归一化）: "
          f"dist(cls1,cls2)={d12:.4f} / dist(cls2,cls3)={d23:.4f} → 比值={ratio:.3f}")
    if ratio > 0.7:
        print("  ⚠  两侧距离相近（比值>0.7），cls2 在特征空间中处于 cls1/cls3 之间，标签混淆风险高")
    else:
        print("  ✓  两侧距离差异较大，cls2 有一侧明显更易混淆")

    # 4. 污染分析
    print("\n" + "─"*70)
    print(" cls2 像素污染分析（基于质心最近邻分类）")
    print("─"*70)
    p = pollution
    print(f"  分析的 cls2 像素数:           {p['n_cls2']:>10,}")
    print(f"  更接近 cls1 质心的像素:       {p['n_closer_to_cls1']:>10,}  "
          f"({p['frac_closer_to_cls1']*100:.1f}%)")
    print(f"  更接近 cls3 质心的像素:       {p['n_closer_to_cls3']:>10,}  "
          f"({p['frac_closer_to_cls3']*100:.1f}%)")
    print(f"  潜在污染像素（偏离cls2）:     {p['n_closer_to_either']:>10,}  "
          f"({p['frac_polluted']*100:.1f}%)")
    print(f"\n  cls2 像素到各质心的平均距离（归一化）:")
    print(f"    → cls1 质心: {p['mean_d_to_cls1']:.4f}")
    print(f"    → cls2 质心: {p['mean_d_to_cls2']:.4f}  （越小说明类内聚集越好）")
    print(f"    → cls3 质心: {p['mean_d_to_cls3']:.4f}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='计算 cls1/cls2/cls3 在 SAR 特征空间中的欧式距离',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data-dir', nargs='+', required=True, metavar='DIR',
                        help='一个或多个 NC 数据集目录（按给定顺序检索）')
    parser.add_argument('--file-list', nargs='+', required=True, metavar='JSON',
                        help='一个或多个 JSON 文件列表')
    parser.add_argument('--features', nargs='+',
                        default=['HH', 'HV'],
                        choices=list(FEATURE_META.keys()),
                        help='参与距离计算的特征 (default: HH HV)')
    parser.add_argument('--max-px-per-scene', type=int, default=5000,
                        help='每个场景每类最多采样的像素数，0=不限制 (default: 5000)')
    parser.add_argument('--max-px-total', type=int, default=500000,
                        help='全局每类最多保留的像素数，防止内存溢出 (default: 500000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (default: 42)')
    parser.add_argument('--ia-correction', action='store_true', default=True,
                        help='对 HH/HV 做入射角线性回归校正后再计算距离（默认开启）')
    parser.add_argument('--no-ia-correction', dest='ia_correction', action='store_false',
                        help='关闭入射角校正')
    parser.add_argument('--ia-ref', type=float, default=None,
                        help='入射角参考值（度），默认取全体像素均值')
    parser.add_argument('--output', default=None,
                        help='可选：将质心统计保存为 CSV')
    args = parser.parse_args()

    np.random.seed(args.seed)

    data_dirs = args.data_dir
    print(f"数据目录: {data_dirs}")

    # 解析特征变量名
    feature_vars = [FEATURE_META[k] for k in args.features]
    print(f"使用特征: {args.features} → {feature_vars}")

    # 读取文件列表
    seen: set[str] = set()
    filenames: list[str] = []
    for json_path in args.file_list:
        with open(json_path, encoding='utf-8') as f:
            for fn in json.load(f):
                if fn not in seen:
                    seen.add(fn)
                    filenames.append(fn)
    print(f"共 {len(filenames)} 个场景")

    # 汇总各类像素
    all_pixels: dict[int, list[np.ndarray]] = {cls: [] for cls in TARGET_CLASSES}
    missing_files = 0

    # 是否需要附带提取 IA（入射角不在用户指定特征里时才需要额外追加）
    ia_var = FEATURE_META['IA']
    append_ia = args.ia_correction and (ia_var not in feature_vars)

    for fn in tqdm(filenames, desc="提取像素"):
        path = resolve_nc_path(fn, data_dirs)
        if path is None:
            missing_files += 1
            print(f"  Skip {fn}: not found in any --data-dir", file=sys.stderr)
            continue
        result = extract_pixels(path, feature_vars,
                                max_px=args.max_px_per_scene,
                                append_ia=append_ia)
        if result is None:
            continue
        for cls, arr in result.items():
            if len(arr) > 0:
                all_pixels[cls].append(arr)

    if missing_files > 0:
        print(f"⚠  共 {missing_files} 个场景未在任何数据目录中找到。", file=sys.stderr)

    # 合并 & 全局下采样
    pixel_arrays: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for cls in TARGET_CLASSES:
        if not all_pixels[cls]:
            print(f"⚠  cls{cls} 没有有效像素，退出。", file=sys.stderr)
            sys.exit(1)
        arr = np.concatenate(all_pixels[cls], axis=0)   # (N_total, F)
        if args.max_px_total > 0 and len(arr) > args.max_px_total:
            idx = np.random.choice(len(arr), args.max_px_total, replace=False)
            arr = arr[idx]
        pixel_arrays[cls] = arr
        counts[cls] = len(arr)

    print(f"\n各类采样像素数: " +
          ", ".join(f"cls{c}={counts[c]:,}" for c in TARGET_CLASSES))

    # 入射角校正（在 z-score 归一化之前做，去除 IA 混淆效应）
    ia_slopes: dict[str, float] | None = None
    ia_ref_used: float | None = None

    if args.ia_correction:
        # IA 在 pixel_arrays 中的列索引
        if append_ia:
            ia_col_idx = -1   # 最后一列是额外追加的 IA
        elif 'IA' in args.features:
            ia_col_idx = args.features.index('IA')
        else:
            ia_col_idx = None

        if ia_col_idx is not None:
            # feature_names 不含 IA（IA 是额外列时），或含 IA（用户显式指定时）
            feat_names_for_corr = [f for f in args.features if f != 'IA']
            pixel_arrays, ia_slopes, ia_ref_used = ia_normalize(
                pixel_arrays,
                feature_names=feat_names_for_corr if append_ia else args.features,
                ia_col_idx=ia_col_idx,
                ia_ref=args.ia_ref,
            )
            # 更新 feature_names：去掉 IA（若用户把 IA 列入 --features，校正后也移除）
            args.features = feat_names_for_corr if 'IA' in args.features else args.features
            counts = {cls: len(pixel_arrays[cls]) for cls in TARGET_CLASSES}
            print(f"入射角校正完成，参考角 = {ia_ref_used:.2f}°")
        else:
            print("⚠  --ia-correction 已开启但未能获取入射角列，跳过校正。", file=sys.stderr)

    # z-score 归一化（基于所有类别合并的均值/std）
    all_combined = np.concatenate(list(pixel_arrays.values()), axis=0)
    global_mean = all_combined.mean(axis=0)    # (F,)
    global_std  = all_combined.std(axis=0).clip(min=1e-8)

    norm_arrays: dict[int, np.ndarray] = {
        cls: (pixel_arrays[cls] - global_mean) / global_std
        for cls in TARGET_CLASSES
    }

    # 计算质心
    centroids_raw  = {cls: pixel_arrays[cls].mean(axis=0) for cls in TARGET_CLASSES}
    centroids_norm = {cls: norm_arrays[cls].mean(axis=0)  for cls in TARGET_CLASSES}
    stds_raw       = {cls: pixel_arrays[cls].std(axis=0)  for cls in TARGET_CLASSES}

    # 计算类别间欧式距离
    pairs = [(1, 2), (2, 3), (1, 3)]
    dist_raw  = {(ci, cj): compute_distance(centroids_raw[ci],  centroids_raw[cj])  for ci, cj in pairs}
    dist_norm = {(ci, cj): compute_distance(centroids_norm[ci], centroids_norm[cj]) for ci, cj in pairs}

    # cls2 污染分析（在归一化空间中进行）
    pollution = analyze_cls2_pollution(norm_arrays[2], centroids_norm)

    # 打印报告
    print_report(
        feature_names=args.features,
        centroids=centroids_raw,
        stds=stds_raw,
        counts=counts,
        dist_raw=dist_raw,
        dist_norm=dist_norm,
        pollution=pollution,
        ia_slopes=ia_slopes,
        ia_ref=ia_ref_used,
    )

    # 可选：保存 CSV
    if args.output:
        import csv
        rows = []
        for cls in TARGET_CLASSES:
            row = {'class': cls}
            for i, fname in enumerate(args.features):
                row[f'{fname}_mean'] = centroids_raw[cls][i]
                row[f'{fname}_std']  = stds_raw[cls][i]
            row['pixel_count'] = counts[cls]
            rows.append(row)

        fieldnames = ['class'] + \
                     [f'{f}_{s}' for f in args.features for s in ('mean', 'std')] + \
                     ['pixel_count']
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        # 追加距离行
        with open(args.output, 'a', newline='', encoding='utf-8') as f:
            f.write('\n# Euclidean distances (normalized)\n')
            for (ci, cj), d in dist_norm.items():
                f.write(f'dist_cls{ci}_cls{cj},{d:.6f}\n')
            f.write(f'# cls2 pollution fraction,{pollution["frac_polluted"]:.4f}\n')

        print(f"\n结果已保存至: {args.output}")


if __name__ == '__main__':
    main()
