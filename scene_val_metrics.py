#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import os.path as osp
from collections import OrderedDict
from typing import Dict, List

import torch
from mmcv import Config
from torchmetrics.functional import f1_score, jaccard_index
from tqdm import tqdm

from functions import class_decider, get_model, slide_inference
from loaders import AI4ArcticChallengeTestDataset, get_variable_options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate per-scene metrics on validation scenes and export filtering artifacts."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/course_report/all.py",
        help="Path to config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="work_dirs/My_DS8/best_model_My_DS8.pth",
        help="Path to .pth checkpoint.",
    )
    parser.add_argument(
        "--val-list",
        type=str,
        default="datalists/train_list.json",
        help="Path to val list JSON (list of scene .nc filenames).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default='/root/autodl-tmp/My_dataset/',
        help="Optional data root that contains the .nc scene files. Overrides config paths.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Override validation dataloader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default= "cpu",
        help="Device, e.g. cuda:0 or cpu. Default: auto.",
    )
    parser.add_argument(
        "--bad-metric",
        type=str,
        default="SOD_mIoU_percent",
        choices=[
            "SOD_f1_percent",
            "SOD_mIoU_percent",
            "SOD_OA_percent",
        ],
        help="Metric used to identify bad scenes.",
    )
    parser.add_argument(
        "--bad-threshold",
        type=float,
        default=60.0,
        help="Scenes with bad_metric < bad_threshold are marked as bad.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="work_dirs/My_DS8/train_scene_metrics.csv",
        help="CSV path for all scene metrics.",
    )
    parser.add_argument(
        "--bad-scenes-csv",
        type=str,
        default="work_dirs/My_DS8/val_bad_scenes.csv",
        help="CSV path for low-score scenes.",
    )
    parser.add_argument(
        "--filtered-val-list",
        type=str,
        default="work_dirs/My_DS8/val_list_filtered.json",
        help="JSON path for filtered val scenes after removing bad scenes.",
    )
    parser.add_argument(
        "--bad-scenes-json",
        type=str,
        default="work_dirs/My_DS8/bad_scenes.json",
        help="JSON path for bad scene filenames.",
    )
    return parser.parse_args()


def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def _round3(x: float) -> float:
    return round(float(x), 3)


def ensure_parent_dir(path: str) -> None:
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_scene_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        scenes = json.load(f)
    if not isinstance(scenes, list):
        raise ValueError(f"Validation list must be a list, got: {type(scenes)}")
    return scenes


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format: expected a dict-like checkpoint.")


def strip_prefix_if_present(state_dict: dict, prefix: str) -> dict:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith(prefix) for k in keys):
        stripped = OrderedDict()
        for k, v in state_dict.items():
            stripped[k[len(prefix):]] = v
        return stripped
    return state_dict


def normalize_state_dict_keys(state_dict: dict) -> dict:
    # Common wrappers: torch.compile ('_orig_mod.'), DataParallel/Distributed ('module.').
    normalized = state_dict
    for p in ["_orig_mod.", "module."]:
        normalized = strip_prefix_if_present(normalized, p)
    return normalized


def resolve_data_root(train_options: dict, val_scenes: List[str], explicit_root: str = None) -> str:
    candidates = []
    if explicit_root:
        candidates.append(explicit_root)
    if train_options.get("path_to_train_data"):
        candidates.append(train_options["path_to_train_data"])
    if train_options.get("path_to_test_data"):
        candidates.append(train_options["path_to_test_data"])
    candidates.append(".")

    scene0 = val_scenes[0] if val_scenes else None
    for root in candidates:
        if not root:
            continue
        root = osp.abspath(root)
        if scene0 is None:
            return root
        if osp.exists(osp.join(root, scene0)):
            return root

    raise FileNotFoundError(
        "Could not locate validation scenes. "
        "Please pass --data-root to the folder containing the .nc files."
    )


def run_scene_inference(net, train_options, inf_x: torch.Tensor, mode: str = "val") -> Dict[str, torch.Tensor]:
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
        if train_options["model_selection"] == "swin":
            output = slide_inference(inf_x, net, train_options, mode)
        else:
            output = net(inf_x)
    return output


def main() -> None:
    args = parse_args()

    cfg = Config.fromfile(args.config)
    train_options = get_variable_options(cfg.train_options)

    val_scenes = read_scene_list(args.val_list)
    if len(val_scenes) == 0:
        raise ValueError("Validation list is empty.")

    data_root = resolve_data_root(train_options, val_scenes, args.data_root)
    train_options["path_to_train_data"] = data_root
    train_options["path_to_test_data"] = data_root
    if args.num_workers is not None:
        train_options["num_workers_val"] = args.num_workers

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            gpu_id = train_options.get("gpu_id", 0)
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")

    net = get_model(train_options, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = extract_state_dict(ckpt)

    load_ok = False
    load_errors = []
    for candidate in [state_dict, normalize_state_dict_keys(state_dict)]:
        try:
            net.load_state_dict(candidate, strict=True)
            load_ok = True
            break
        except RuntimeError as e:
            load_errors.append(str(e))

    if not load_ok:
        err = "\n\n".join(load_errors)
        raise RuntimeError(
            "Failed to load checkpoint even after key normalization. "
            "Please verify config/checkpoint match.\n\n" + err
        )

    net.eval()

    dataset = AI4ArcticChallengeTestDataset(
        options=train_options,
        files=val_scenes,
        mode="train",
    )
    asid_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=train_options["num_workers_val"],
        shuffle=False,
    )

    rows = []
    charts = ["SOD"]
    if "SOD" not in train_options["n_classes"]:
        raise KeyError("SOD not found in train_options['n_classes']")
    n_classes = train_options["n_classes"]

    fieldnames = [
        "scene",
        "SOD_valid_pixels",
        "SOD_f1_percent",
        "SOD_mIoU_percent",
        "SOD_OA_percent",
    ]
    ensure_parent_dir(args.output_csv)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as metrics_f:
        writer = csv.DictWriter(metrics_f, fieldnames=fieldnames)
        writer.writeheader()
        metrics_f.flush()

        for inf_x, inf_y, cfv_masks, _tfv_mask, scene_name, original_size in tqdm(
            asid_loader,
            total=len(val_scenes),
            desc="Evaluating scenes",
            colour="green",
        ):
            scene_name = osp.splitext(scene_name)[0]
            inf_x = inf_x.to(device, non_blocking=True)

            output = run_scene_inference(net, train_options, inf_x, mode="val")

            for chart in charts:
                masks_int = cfv_masks[chart].to(torch.uint8)
                masks_int = torch.nn.functional.interpolate(
                    masks_int.unsqueeze(0).unsqueeze(0),
                    size=original_size,
                    mode="nearest",
                ).squeeze().squeeze()
                cfv_masks[chart] = torch.gt(masks_int, 0)

            if train_options["down_sample_scale"] != 1:
                for chart in charts:
                    if output[chart].size(3) == 1:
                        output[chart] = output[chart].permute(0, 3, 1, 2)
                        output[chart] = torch.nn.functional.interpolate(
                            output[chart], size=original_size, mode="nearest"
                        )
                        output[chart] = output[chart].permute(0, 2, 3, 1)
                    else:
                        output[chart] = torch.nn.functional.interpolate(
                            output[chart], size=original_size, mode="nearest"
                        )

                    inf_y[chart] = torch.nn.functional.interpolate(
                        inf_y[chart].unsqueeze(dim=0).unsqueeze(dim=0),
                        size=original_size,
                        mode="nearest",
                    ).squeeze()

            row = {"scene": scene_name}

            for chart in charts:
                pred = class_decider(output[chart], train_options, chart).detach().long()
                true = inf_y[chart].long().to(device, non_blocking=True)
                valid = (~cfv_masks[chart]).to(device)

                pred_flat = pred[valid]
                true_flat = true[valid]
                valid_pixels = int(true_flat.numel())
                row[f"{chart}_valid_pixels"] = valid_pixels

                if valid_pixels == 0:
                    row[f"{chart}_f1_percent"] = float("nan")
                    row[f"{chart}_mIoU_percent"] = float("nan")
                    row[f"{chart}_OA_percent"] = float("nan")
                    continue

                f1 = f1_score(
                    target=true_flat,
                    preds=pred_flat,
                    average="weighted",
                    task="multiclass",
                    num_classes=n_classes[chart],
                )
                miou = jaccard_index(
                    target=true_flat,
                    preds=pred_flat,
                    task="multiclass",
                    num_classes=n_classes[chart],
                )
                oa = (true_flat == pred_flat).float().mean()

                f1_percent = _round3(_to_float(f1) * 100.0)
                miou_percent = _round3(_to_float(miou) * 100.0)
                oa_percent = _round3(_to_float(oa) * 100.0)

                row[f"{chart}_f1_percent"] = f1_percent
                row[f"{chart}_mIoU_percent"] = miou_percent
                row[f"{chart}_OA_percent"] = oa_percent

            rows.append(row)
            writer.writerow(row)
            metrics_f.flush()

    rows.sort(key=lambda x: x["scene"])

    bad_metric = args.bad_metric
    bad_scenes_rows = [r for r in rows if r.get(bad_metric, float("nan")) < args.bad_threshold]
    bad_scenes_rows.sort(key=lambda r: r.get(bad_metric, float("inf")))

    ensure_parent_dir(args.bad_scenes_csv)
    with open(args.bad_scenes_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bad_scenes_rows)

    bad_scene_stems = {r["scene"] for r in bad_scenes_rows}
    bad_scene_files = [s for s in val_scenes if osp.splitext(osp.basename(s))[0] in bad_scene_stems]
    filtered_val_scenes = [s for s in val_scenes if s not in bad_scene_files]

    ensure_parent_dir(args.bad_scenes_json)
    with open(args.bad_scenes_json, "w", encoding="utf-8") as f:
        json.dump(bad_scene_files, f, indent=2, ensure_ascii=False)

    ensure_parent_dir(args.filtered_val_list)
    with open(args.filtered_val_list, "w", encoding="utf-8") as f:
        json.dump(filtered_val_scenes, f, indent=2, ensure_ascii=False)

    print("Per-scene evaluation completed.")
    print(f"All scene metrics CSV: {args.output_csv}")
    print(f"Bad scenes CSV: {args.bad_scenes_csv}")
    print(f"Bad scenes JSON: {args.bad_scenes_json}")
    print(f"Filtered val list JSON: {args.filtered_val_list}")
    print(f"Bad scene criterion: {bad_metric} < {args.bad_threshold}")
    print(f"Removed scenes: {len(bad_scene_files)} / {len(val_scenes)}")


if __name__ == "__main__":
    main()
