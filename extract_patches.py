"""
Patch extraction script for pure SOD patches from MyDS NC files.

Strategy:
- Sliding window (stride=128) over each scene
- Accept a patch only if:
    1. global_valid_mask valid ratio >= VALID_MASK_THRESHOLD (default 0.95)
    2. All valid (non-black-border, non-masked) SOD pixels are the same class (100% purity)
- Collect up to M=1000 patches per SOD class (0-5, NOT merged)
- Stop collecting for a class once M is reached; stop entirely when all classes done or all scenes exhausted
- Save features as float32 .npy, SOD labels as uint8 .npy
- Save metadata.json recording source scene and crop coordinates for each patch

Output structure:
    patch_dataset/
    ├── class_0/   scene_xxx_r128_c256_feat.npy  (shape [6,256,256])
    │              scene_xxx_r128_c256_sod.npy   (shape [256,256])
    ├── class_1/
    ...
    ├── class_5/
    └── metadata.json

Scene-level split note:
    Only extract from train_list.json scenes.
    When training, split by *scene* (NC filename), not by patch,
    so that val patches come from scenes the model has never seen.
"""

import json
import os
import os.path as osp

import numpy as np
import xarray as xr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NC_DIR = r'F:\ZJU\11_Ice\dataset_create\dataset_nc_zscore_new'
TRAIN_LIST = 'datalists/train_list.json'
OUTPUT_DIR = r'F:\ZJU\11_Ice\dataset_create\patch_dataset'

PATCH_SIZE = 256
STRIDE = 128
M = 1000            # target patches per class (total)
MAX_PER_SCENE = 50  # max patches per class per scene (forces scene diversity)
N_CLASSES = 6       # SOD classes 0-5, NOT merged (no remap applied)
SOD_MASK = 255      # fill/ignore value in SOD label
VALID_MASK_THRESHOLD = 0.95  # min fraction of valid (non-black-border) pixels

FEATURE_VARS = [
    'nersc_sar_primary',
    'nersc_sar_secondary',
    'sar_incidenceangle',
    'glcm_sigma0_hh_contrast',
    'glcm_sigma0_hh_dissimilarity',
    'glcm_sigma0_hh_homogeneity',
]
SOD_VAR = 'SOD'
VALID_MASK_VAR = 'global_valid_mask'

VALID_CLASSES = set(range(N_CLASSES))  # {0,1,2,3,4,5}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    with open(TRAIN_LIST) as f:
        file_list = json.load(f)

    counts = {c: 0 for c in range(N_CLASSES)}
    done = {c: False for c in range(N_CLASSES)}

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for cls in range(N_CLASSES):
        os.makedirs(osp.join(OUTPUT_DIR, f'class_{cls}'), exist_ok=True)

    metadata = []

    # Multi-pass loop: each pass allows up to `current_cap` patches per class
    # per scene. After a full pass over all scenes, if some classes are still
    # incomplete, raise the cap by PASS_INCREMENT and try again. This ensures
    # patches are spread across as many scenes as possible before any single
    # scene contributes more.
    PASS_INCREMENT = MAX_PER_SCENE
    current_cap = MAX_PER_SCENE
    pass_num = 1

    while not all(done.values()):
        tqdm.write(f'\n=== Pass {pass_num} (per-scene cap per class: {current_cap}) ===')
        counts_before_pass = dict(counts)

        for filename in tqdm(file_list, desc=f'Pass {pass_num}'):
            if all(done.values()):
                break

            nc_path = osp.join(NC_DIR, filename)
            if not osp.exists(nc_path):
                tqdm.write(f'[WARN] File not found, skipping: {nc_path}')
                continue

            scene = xr.open_dataset(nc_path, engine='h5netcdf', mask_and_scale=False)

            # Only load the two lightweight mask/label arrays upfront for scanning.
            # Feature data is large (~1.8 GiB per scene) and loaded lazily per patch.
            sod_arr = scene[SOD_VAR].values          # (H, W)
            valid_mask = scene[VALID_MASK_VAR].values # (H, W)

            H, W = sod_arr.shape
            scene_stem = osp.splitext(filename)[0]

            # Per-scene counters: how many patches this scene contributes this pass.
            scene_counts = {c: 0 for c in range(N_CLASSES)}
            # Effective cap for this scene in this pass: only collect up to
            # current_cap minus what was already collected in previous passes.
            # (Patches saved in pass 1 already exist on disk; don't re-save.)
            prev_cap = current_cap - PASS_INCREMENT  # cap used in all prior passes
            scene_cap = {c: current_cap - prev_cap for c in range(N_CLASSES)}

            n_rows = (H - PATCH_SIZE) // STRIDE + 1
            n_cols = (W - PATCH_SIZE) // STRIDE + 1

            for ri in range(n_rows):
                for ci in range(n_cols):
                    r = ri * STRIDE
                    c = ci * STRIDE

                    vm_patch = valid_mask[r:r + PATCH_SIZE, c:c + PATCH_SIZE]

                    # --- Filter 1: valid pixel ratio ---
                    valid_ratio = vm_patch.mean()
                    if valid_ratio < VALID_MASK_THRESHOLD:
                        continue

                    sod_patch = sod_arr[r:r + PATCH_SIZE, c:c + PATCH_SIZE]

                    # Valid SOD pixels: inside valid mask AND not SOD fill value
                    valid_sod_mask = (vm_patch == 1) & (sod_patch != SOD_MASK)
                    valid_sod_vals = sod_patch[valid_sod_mask]

                    if len(valid_sod_vals) == 0:
                        continue

                    # --- Filter 2: 100% purity among valid SOD pixels ---
                    unique_vals = np.unique(valid_sod_vals)
                    if len(unique_vals) != 1:
                        continue

                    cls = int(unique_vals[0])
                    if cls not in VALID_CLASSES:
                        continue  # unexpected value, skip

                    if done[cls]:
                        continue
                    if scene_counts[cls] >= scene_cap[cls]:
                        continue

                    # --- Save patch ---
                    # Read only this patch region from disk (lazy xarray indexing).
                    feat_patch = np.stack(
                        [scene[v].values[r:r + PATCH_SIZE, c:c + PATCH_SIZE]
                         for v in FEATURE_VARS], axis=0
                    ).astype(np.float32)  # (6, 256, 256)
                    patch_id = f'{scene_stem}_r{r}_c{c}'
                    feat_path = osp.join(OUTPUT_DIR, f'class_{cls}', f'{patch_id}_feat.npy')
                    lbl_path  = osp.join(OUTPUT_DIR, f'class_{cls}', f'{patch_id}_sod.npy')

                    np.save(feat_path, feat_patch)
                    np.save(lbl_path, sod_patch.astype(np.uint8))

                    metadata.append({
                        'scene':    filename,
                        'row':      r,
                        'col':      c,
                        'class':    cls,
                        'feat':     feat_path,
                        'label':    lbl_path,
                    })

                    counts[cls] += 1
                    scene_counts[cls] += 1
                    if counts[cls] >= M:
                        done[cls] = True
                        tqdm.write(f'  [Class {cls}] reached {M} patches.')

            scene.close()

        # Check if this pass made any progress; if not, no more patches available.
        made_progress = any(counts[c] > counts_before_pass[c] for c in range(N_CLASSES))
        if not made_progress:
            tqdm.write('No new patches found in this pass. Stopping.')
            break

        current_cap += PASS_INCREMENT
        pass_num += 1

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print('\n=== Extraction Summary ===')
    for cls in range(N_CLASSES):
        status = 'COMPLETE' if done[cls] else f'INCOMPLETE — only {counts[cls]}/{M} patches found'
        print(f'  Class {cls}: {counts[cls]:4d} patches  [{status}]')

    meta_path = osp.join(OUTPUT_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'\nMetadata saved to: {meta_path}')
    print(f'Total patches: {sum(counts.values())}')


if __name__ == '__main__':
    main()
