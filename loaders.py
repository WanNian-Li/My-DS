
# -- Built-in modules -- #
import os
import os.path as osp
import csv
import math
from tqdm import tqdm

# -- Third-party modules -- #
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from scipy.ndimage import binary_erosion as _scipy_binary_erosion

# -- Proprietary modules -- #
from functions import rand_bbox


def _erode_sod_boundaries(sod_patch, erosion_iters, ignore_val=255):
    """将SOD标签各类别的边界像素设为 ignore_val，只保留类别核心区域参与训练。

    CIS冰蛋图polygon与Sentinel-1 SAR存在时空错位，交界处像素标签最不可信。
    对每个类别的区域向内腐蚀 erosion_iters 像素，被腐蚀掉的边界部分设为 ignore_val。

    Parameters
    ----------
    sod_patch : ndarray, shape (H, W)
        SOD标签数组，ignore_val 表示无效像素。
    erosion_iters : int
        向内腐蚀的像素数，对应CIS对齐误差估计（建议3-7）。
    ignore_val : int
        无效像素的标记值，默认255。

    Returns
    -------
    result : ndarray, shape (H, W)
        边界像素已替换为 ignore_val 的标签数组。
    """
    result = sod_patch.copy()
    valid_mask = sod_patch != ignore_val
    for cls_id in np.unique(sod_patch[valid_mask]):
        cls_mask = (sod_patch == cls_id)
        eroded = _scipy_binary_erosion(cls_mask, iterations=erosion_iters)
        boundary = cls_mask & ~eroded
        result[boundary] = ignore_val
    return result


def _month_sin_cos(filename):
    """Extract month from scene filename and return (sin, cos) cyclic encoding.

    Filename format: S1A_EW_GRDM_1SDH_YYYYMMDDTHHMMSS_...nc
    Month digits are at filename[21:23].
    """
    month = int(filename[21:23])
    angle = 2 * math.pi * month / 12
    return math.sin(angle), math.cos(angle)


class AI4ArcticChallengeDataset(Dataset):
    """Pytorch dataset for loading batches of patches of scenes from the ASID
    V2 data set."""

    def __init__(self, options, files, do_transform=False):
        self.options = options
        self.files = files
        self.do_transform = do_transform

        # If Downscaling, down sample data and put in on memory
        if (self.options['down_sample_scale'] == 1):
            self.downsample = False
        else:
            self.downsample = True

        if self.downsample:
            self.scenes = []
            for file in tqdm(self.files):
                scene = xr.open_dataset(os.path.join(
                    self.options['path_to_train_data'], file), engine='h5netcdf', mask_and_scale=False)

                temp_scene = scene[self.options['full_variables']].to_array()
                temp_scene = torch.from_numpy(np.expand_dims(temp_scene, 0))
                temp_scene = torch.nn.functional.interpolate(temp_scene,
                                                             size=(temp_scene.size(2)//self.options['down_sample_scale'],
                                                                   temp_scene.size(3)//self.options['down_sample_scale']),
                                                             mode=self.options['loader_downsampling'])

                if temp_scene.size(2) < self.options['patch_size']:
                    height_pad = self.options['patch_size'] - temp_scene.size(2) + 1
                else:
                    height_pad = 0

                if temp_scene.size(3) < self.options['patch_size']:
                    width_pad = self.options['patch_size'] - temp_scene.size(3) + 1
                else:
                    width_pad = 0

                if height_pad > 0 or width_pad > 0:
                    temp_scene_y = torch.nn.functional.pad(
                        temp_scene[:, :len(self.options['charts'])], (0, width_pad, 0, height_pad), mode='constant', value=255)
                    temp_scene_x = torch.nn.functional.pad(
                        temp_scene[:, len(self.options['charts']):], (0, width_pad, 0, height_pad), mode='constant', value=0)
                    temp_scene = torch.cat((temp_scene_y, temp_scene_x), dim=1)

                temp_scene = torch.squeeze(temp_scene)
                self.scenes.append(temp_scene)

        # Precompute global_valid_mask channel index for black border filtering during training
        if 'global_valid_mask' in self.options['train_variables']:
            self.global_valid_mask_idx = (len(self.options['charts']) +
                                          list(self.options['train_variables']).index('global_valid_mask'))
        else:
            self.global_valid_mask_idx = None

        # Channel numbers in patches, includes reference channel.
        self.patch_c = len(
            self.options['train_variables']) + len(self.options['charts'])

        # Kept for backward compatibility; epoch-level logging is handled in quickstart.py.
        self.patch_log = []
        self._target_chart = self.options.get('target_chart', 'SOD')
        self._target_chart_idx = self.options['charts'].index(self._target_chart)

        # Two-phase sampling: pre-build class location bank if enabled.
        if self.options.get('two_phase_sampling', False):
            self.class_location_bank = self._build_class_location_bank()
        else:
            self.class_location_bank = {}

    def __len__(self):
        return self.options['epoch_len']

    # ------------------------------------------------------------------ #
    # Two-phase sampling helpers                                           #
    # ------------------------------------------------------------------ #

    def _build_class_location_bank(self):
        """Scan all scenes at coarse stride to build a class→location index.

        For each valid patch crop position, records (scene_id, row, col) under
        every mapped class present in that patch. Used by Phase 2 of two-phase
        sampling to efficiently retrieve patches containing rare classes.

        Stride = patch_size * class_bank_stride_factor (default 0.5).

        Returns
        -------
        bank : dict
            {class_id (int): [(scene_id, row, col), ...]}
        """
        stride_factor = self.options.get('class_bank_stride_factor', 0.5)
        stride = max(1, int(self.options['patch_size'] * stride_factor))
        patch_size = self.options['patch_size']
        fill_val = self.options['class_fill_values'][self._target_chart]

        bank = {}

        for scene_id, file in enumerate(tqdm(self.files, desc='Building class location bank')):
            if self.downsample:
                label_map = self.scenes[scene_id][self._target_chart_idx].numpy().copy()
            else:
                try:
                    scene = xr.open_dataset(
                        os.path.join(self.options['path_to_train_data'], file),
                        engine='h5netcdf', mask_and_scale=False)
                    label_map = scene[self._target_chart].values.copy()
                    scene.close()
                except Exception:
                    continue

            # Apply same label remapping as random_crop so bank indexes mapped classes.
            if self._target_chart == 'SOD':
                valid = label_map != 255
                label_map[valid & (label_map >= 2)] -= 1
            elif self._target_chart == 'FLOE':
                label_map[(label_map >= 1) & (label_map <= 3)] = 255
                label_map[label_map == 6] = 255
                label_map[label_map == 4] = 1
                label_map[label_map == 5] = 2

            H, W = label_map.shape
            for row in range(0, H - patch_size, stride):
                for col in range(0, W - patch_size, stride):
                    patch_label = label_map[row:row + patch_size, col:col + patch_size]
                    valid_pixels = patch_label[patch_label != fill_val]
                    if len(valid_pixels) == 0:
                        continue
                    for cls in np.unique(valid_pixels):
                        cls = int(cls)
                        if cls not in bank:
                            bank[cls] = []
                        bank[cls].append((scene_id, row, col))

        summary = ', '.join(f'cls{c}:{len(v)}' for c, v in sorted(bank.items()))
        print(f'Class location bank built: {summary}')
        return bank

    def _crop_at(self, scene_id, row, col):
        """Crop a patch at a fixed (scene_id, row, col).

        Applies the same label remapping, erosion, and validity checks as
        random_crop / random_crop_downsample.

        Returns
        -------
        (x_patch, y_patch) or (None, None) if the location is invalid.
        """
        patch_size = self.options['patch_size']
        patch = np.zeros((len(self.options['full_variables']), patch_size, patch_size))
        fill_val = self.options['class_fill_values'][self._target_chart]

        if self.downsample:
            if self.global_valid_mask_idx is not None:
                vm = self.scenes[scene_id][
                    self.global_valid_mask_idx,
                    row:row + patch_size, col:col + patch_size].numpy()
                if float(np.mean(vm)) < self.options.get('valid_mask_threshold', 0.5):
                    return None, None

            label_patch = self.scenes[scene_id][
                self._target_chart_idx,
                row:row + patch_size, col:col + patch_size].numpy()
            if np.sum(label_patch != fill_val) <= 1:
                return None, None

            patch[:] = self.scenes[scene_id][
                :, row:row + patch_size, col:col + patch_size].numpy()
        else:
            try:
                scene = xr.open_dataset(
                    os.path.join(self.options['path_to_train_data'], self.files[scene_id]),
                    engine='h5netcdf', mask_and_scale=False)
            except Exception:
                return None, None

            if 'global_valid_mask' in self.options['train_variables']:
                vm = scene['global_valid_mask'].isel(
                    y=slice(row, row + patch_size),
                    x=slice(col, col + patch_size)).values
                if float(np.mean(vm)) < self.options.get('valid_mask_threshold', 0.5):
                    scene.close()
                    return None, None

            label_patch = scene[self._target_chart].isel(
                y=slice(row, row + patch_size),
                x=slice(col, col + patch_size)).values
            if np.sum(label_patch != fill_val) <= 1:
                scene.close()
                return None, None

            patch[:] = scene[self.options['full_variables']].isel(
                y=range(row, row + patch_size),
                x=range(col, col + patch_size)).to_array().values
            scene.close()

        # Apply same label remapping as random_crop.
        if self._target_chart == 'SOD':
            sod_ch = patch[self._target_chart_idx]
            valid = sod_ch != 255
            sod_ch[valid & (sod_ch >= 2)] -= 1
        elif self._target_chart == 'FLOE':
            floe_ch = patch[self._target_chart_idx]
            floe_ch[(floe_ch >= 1) & (floe_ch <= 3)] = 255
            floe_ch[floe_ch == 6] = 255
            floe_ch[floe_ch == 4] = 1
            floe_ch[floe_ch == 5] = 2

        _erode_iters = self.options.get('boundary_erosion_iters', 0)
        if _erode_iters > 0:
            patch[self._target_chart_idx] = _erode_sod_boundaries(
                patch[self._target_chart_idx], erosion_iters=_erode_iters)

        x_patch = torch.from_numpy(
            patch[len(self.options['charts']):, :]).type(torch.float).unsqueeze(0)
        y_patch = torch.from_numpy(
            patch[:len(self.options['charts']), :, :]).unsqueeze(0)
        return x_patch, y_patch

    def _augment_x(self, x_patch, scene_id):
        """Append month encoding and/or HH-HV polarization ratio channels to x_patch."""
        if self.options.get('month_encoding', False):
            s, c = _month_sin_cos(self.files[scene_id])
            H, W = x_patch.shape[-2], x_patch.shape[-1]
            month_ch = torch.tensor([s, c], dtype=torch.float).view(1, 2, 1, 1).expand(1, 2, H, W)
            x_patch = torch.cat([x_patch, month_ch], dim=1)
        if self.options.get('pol_ratio_channel', False):
            pol_ratio = x_patch[:, 0:1, :, :] - x_patch[:, 1:2, :, :]
            x_patch = torch.cat([x_patch, pol_ratio], dim=1)
        return x_patch

    def random_crop(self, scene):
        """
        Perform random cropping in scene.

        Parameters
        ----------
        scene :
            Xarray dataset; a MyDS scene with dimensions ('y', 'x').

        Returns
        -------
        x_patch :
            torch array with shape (len(train_variables),
            patch_height, patch_width). None if empty patch.
        y_patch :
            torch array with shape (len(charts),
            patch_height, patch_width). None if empty patch.
        """
        patch = np.zeros((len(self.options['full_variables']),
                          self.options['patch_size'],
                          self.options['patch_size']))

        # Get random index to crop from.
        row_rand = np.random.randint(
            low=0, high=scene[self._target_chart].values.shape[0]
            - self.options['patch_size'])
        col_rand = np.random.randint(
            low=0, high=scene[self._target_chart].values.shape[1]
            - self.options['patch_size'])

        # Discard patches with too many black border (georrectification artifact) pixels.
        if 'global_valid_mask' in self.options['train_variables']:
            valid_mask_patch = scene['global_valid_mask'].isel(
                y=slice(row_rand, row_rand + self.options['patch_size']),
                x=slice(col_rand, col_rand + self.options['patch_size'])).values
            valid_ratio = float(np.mean(valid_mask_patch))
            if valid_ratio < self.options.get('valid_mask_threshold', 0.5):
                return None, None

        target_patch = scene[self._target_chart].isel(
            y=slice(row_rand, row_rand + self.options['patch_size']),
            x=slice(col_rand, col_rand + self.options['patch_size'])).values
        if np.sum(target_patch != self.options['class_fill_values'][self._target_chart]) > 1:

            # Crop all full-resolution variables (MyDS: all variables at same 80m resolution).
            patch[0:len(self.options['full_variables']), :, :] = \
                scene[self.options['full_variables']].isel(
                y=range(row_rand, row_rand + self.options['patch_size']),
                x=range(col_rand, col_rand + self.options['patch_size'])).to_array().values

            # SOD-specific: remap labels in-place: merge old class 2 (young ice) into
            # class 1 (new/young ice). All old labels >=2 (excluding mask=255) shift down by 1.
            if self._target_chart == 'SOD':
                sod_ch = patch[self._target_chart_idx]
                valid = sod_ch != 255
                sod_ch[valid & (sod_ch >= 2)] -= 1

            # FLOE-specific label remapping:
            #   1(冰块), 2(小浮冰), 3(中浮冰), 6(冰山) → 255 (忽略)
            #   4(大浮冰) → 1, 5(巨浮冰) → 2
            elif self._target_chart == 'FLOE':
                floe_ch = patch[self._target_chart_idx]
                floe_ch[(floe_ch >= 1) & (floe_ch <= 3)] = 255
                floe_ch[floe_ch == 6] = 255
                floe_ch[floe_ch == 4] = 1
                floe_ch[floe_ch == 5] = 2

            # Boundary erosion: mask out class-boundary pixels that are most likely
            # mislabeled due to spatial misalignment between CIS ice charts and SAR.
            _erode_iters = self.options.get('boundary_erosion_iters', 0)
            if _erode_iters > 0:
                patch[self._target_chart_idx] = _erode_sod_boundaries(
                    patch[self._target_chart_idx], erosion_iters=_erode_iters)

            x_patch = torch.from_numpy(
                patch[len(self.options['charts']):, :]).type(torch.float).unsqueeze(0)
            y_patch = torch.from_numpy(patch[:len(self.options['charts']), :, :]).unsqueeze(0)

        # In case patch does not contain any valid pixels - return None.
        else:
            x_patch = None
            y_patch = None

        return x_patch, y_patch

    def random_crop_downsample(self, idx):
        """
        Perform random cropping on a pre-downsampled scene (stored in self.scenes).

        Parameters
        ----------
        idx :
            Index from self.files to parse.

        Returns
        -------
        x_patch, y_patch :
            Torch tensors, or (None, None) if patch is invalid.
        """

        patch = np.zeros((len(self.options['full_variables']),
                          self.options['patch_size'],
                          self.options['patch_size']))

        # Get random index to crop from.
        row_rand = np.random.randint(
            low=0, high=self.scenes[idx].size(1)
            - self.options['patch_size'])
        col_rand = np.random.randint(
            low=0, high=self.scenes[idx].size(2)
            - self.options['patch_size'])

        # Discard patches with too many black border pixels using global_valid_mask.
        if self.global_valid_mask_idx is not None:
            valid_mask_patch = self.scenes[idx][
                self.global_valid_mask_idx,
                row_rand: row_rand + self.options['patch_size'],
                col_rand: col_rand + self.options['patch_size']].numpy()
            valid_ratio = float(np.mean(valid_mask_patch))
            if valid_ratio < self.options.get('valid_mask_threshold', 0.5):
                return None, None

        # Discard patches with too many label fill (masked) pixels.
        if np.sum(self.scenes[idx][self._target_chart_idx,
                                   row_rand: row_rand + self.options['patch_size'],
                                   col_rand: col_rand + self.options['patch_size']].numpy()
                  != self.options['class_fill_values'][self._target_chart]) > 1:

            patch[0:len(self.options['full_variables']), :, :] = \
                self.scenes[idx][:, row_rand:row_rand + int(self.options['patch_size']),
                                 col_rand:col_rand + int(self.options['patch_size'])].numpy()

            # SOD-specific: remap labels in-place: merge old class 2 (young ice) into
            # class 1 (new/young ice). All old labels >=2 (excluding mask=255) shift down by 1.
            if self._target_chart == 'SOD':
                sod_ch = patch[self._target_chart_idx]
                valid = sod_ch != 255
                sod_ch[valid & (sod_ch >= 2)] -= 1

            # FLOE-specific label remapping:
            #   1(冰块), 2(小浮冰), 3(中浮冰), 6(冰山) → 255 (忽略)
            #   4(大浮冰) → 1, 5(巨浮冰) → 2
            elif self._target_chart == 'FLOE':
                floe_ch = patch[self._target_chart_idx]
                floe_ch[(floe_ch >= 1) & (floe_ch <= 3)] = 255
                floe_ch[floe_ch == 6] = 255
                floe_ch[floe_ch == 4] = 1
                floe_ch[floe_ch == 5] = 2

            # Boundary erosion: mask out class-boundary pixels that are most likely
            # mislabeled due to spatial misalignment between CIS ice charts and SAR.
            _erode_iters = self.options.get('boundary_erosion_iters', 0)
            if _erode_iters > 0:
                patch[self._target_chart_idx] = _erode_sod_boundaries(
                    patch[self._target_chart_idx], erosion_iters=_erode_iters)

            x_patch = torch.from_numpy(
                patch[len(self.options['charts']):, :]).type(torch.float).unsqueeze(0)
            y_patch = torch.from_numpy(patch[:len(self.options['charts']), :, :]).unsqueeze(0)

        # In case patch does not contain any valid pixels - return None.
        else:
            x_patch = None
            y_patch = None

        return x_patch, y_patch

    def prep_dataset(self, x_patches, y_patches):
        """
        Convert patches from 4D numpy array to 4D torch tensor.

        Parameters
        ----------
        x_patches : ndarray
            Patches sampled from ASID3 ready-to-train challenge dataset scenes [PATCH, CHANNEL, H, W] containing only the trainable variables.
        y_patches : ndarray
            Patches sampled from ASID3 ready-to-train challenge dataset scenes [PATCH, CHANNEL, H, W] contrainng only the targets.

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y : Dict
            Dictionary with 3D torch tensors for each chart; reference data for training data x.
        """

        # Convert training data to tensor float.
        x = x_patches.type(torch.float)

        # Store charts in y dictionary.

        y = {}
        for idx, chart in enumerate(self.options['charts']):
            y[chart] = y_patches[:, idx].type(torch.long)

        return x, y
    
    def transform(self, x_patch, y_patch):
        data_aug_options = self.options['data_augmentations']
        if torch.rand(1) < data_aug_options['Random_h_flip']:
            x_patch = TF.hflip(x_patch)
            y_patch = TF.hflip(y_patch)

        if torch.rand(1) < data_aug_options['Random_v_flip']:
            x_patch = TF.vflip(x_patch)
            y_patch = TF.vflip(y_patch)

        assert (data_aug_options['Random_rotation'] <= 180)
        if data_aug_options['Random_rotation'] != 0 and \
                torch.rand(1) < data_aug_options['Random_rotation_prob']:
            random_degree = np.random.randint(-data_aug_options['Random_rotation'],
                                                data_aug_options['Random_rotation']
                                                )
        else:
            random_degree = 0

        scale_diff = data_aug_options['Random_scale'][1] - \
            data_aug_options['Random_scale'][0]
        assert (scale_diff >= 0)
        if scale_diff != 0 and torch.rand(1) < data_aug_options['Random_scale_prob']:
            random_scale = np.random.rand()*(data_aug_options['Random_scale'][1] -
                                                data_aug_options['Random_scale'][0]) +\
                data_aug_options['Random_scale'][0]
        else:
            # random_scale = data_aug_options['Random_scale'][1]
            random_scale = 1.0 

        x_patch = TF.affine(x_patch, angle=random_degree, translate=(0, 0),
                            shear=0, scale=random_scale, fill=0)
        y_patch = TF.affine(y_patch, angle=random_degree, translate=(0, 0),
                            shear=0, scale=random_scale, fill=255)
        
        return x_patch, y_patch

    def __getitem__(self, idx):
        """
        Get batch. Function required by Pytorch dataset.

        When two_phase_sampling is enabled:
          Phase 1 — fills base_ratio of the batch with random crops (same filters
                    as before).
          Phase 2 — inspects the Phase 1 pixel distribution, computes per-class
                    deficit against target_class_ratio, then draws compensatory
                    patches from class_location_bank to cover the shortfall.

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y : Dict
            Dictionary with 3D torch tensors for each chart; reference data for training data x.
        """
        _n_channels = (len(self.options['train_variables'])
                       + (2 if self.options.get('month_encoding', False) else 0)
                       + (1 if self.options.get('pol_ratio_channel', False) else 0))
        x_patches = torch.zeros((self.options['batch_size'], _n_channels,
                                 self.options['patch_size'], self.options['patch_size']))
        y_patches = torch.zeros((self.options['batch_size'], len(self.options['charts']),
                                 self.options['patch_size'], self.options['patch_size']))
        sample_n = 0

        _two_phase = (self.options.get('two_phase_sampling', False)
                      and len(self.class_location_bank) > 0)
        base_n = (max(1, int(self.options['batch_size'] * self.options.get('base_ratio', 0.6)))
                  if _two_phase else self.options['batch_size'])

        # ------------------------------------------------------------------ #
        # Phase 1 — random sampling (same logic as before).                   #
        # ------------------------------------------------------------------ #
        while sample_n < base_n:
            scene_id = np.random.randint(low=0, high=len(self.files), size=1).item()
            try:
                if self.downsample:
                    x_patch, y_patch = self.random_crop_downsample(scene_id)
                else:
                    scene = xr.open_dataset(os.path.join(
                        self.options['path_to_train_data'], self.files[scene_id]),
                        engine='h5netcdf', mask_and_scale=False)
                    x_patch, y_patch = self.random_crop(scene)
                    scene.close()
            except FileNotFoundError:
                print(f"File {self.files[scene_id]} not found. Skipping scene.")
                continue
            except Exception:
                if self.downsample:
                    print(f"Cropping in {self.files[scene_id]} failed. "
                          f"Scene size: {self.scenes[scene_id][0].shape}. Skipping.")
                    continue
                else:
                    print(f"Cropping in {self.files[scene_id]} failed. "
                          f"Scene size: {scene[self._target_chart].shape}. Skipping.")
                    scene.close()
                    continue

            if x_patch is not None:
                x_patch = self._augment_x(x_patch, scene_id)
                target_flat = y_patch[0, self._target_chart_idx].numpy().flatten()

                # Filter 1 — Label coverage.
                sod_invalid_max = self.options.get('sod_invalid_max_ratio', 1.0)
                if sod_invalid_max < 1.0:
                    if float((target_flat == 255).sum()) / len(target_flat) > sod_invalid_max:
                        continue

                # Filter 2 — Water patch rejection.
                water_max_ratio = self.options.get('water_patch_max_ratio', 1.0)
                water_reject_prob = self.options.get('water_rejection_prob', 0.0)
                if water_reject_prob > 0.0 and water_max_ratio < 1.0:
                    valid_mask = target_flat != 255
                    valid_count = valid_mask.sum()
                    if valid_count > 0:
                        water_ratio = float((target_flat[valid_mask] == 0).sum()) / valid_count
                        if water_ratio > water_max_ratio and np.random.rand() < water_reject_prob:
                            continue

                # Filter 3 — Rare-class weighted sampling.
                _rare_classes = self.options.get('rare_sampling_classes', [])
                _rare_alpha = self.options.get('rare_sampling_alpha', 0.0)
                if _rare_classes and _rare_alpha > 0.0:
                    _valid_target = target_flat[target_flat != 255]
                    if len(_valid_target) > 0:
                        _rare_count = float(sum((_valid_target == c).sum() for c in _rare_classes))
                        _rare_frac = _rare_count / len(_valid_target)
                        _accept_prob = (1.0 - _rare_alpha) + _rare_alpha * _rare_frac
                        if np.random.rand() > _accept_prob:
                            continue

                if self.do_transform:
                    x_patch, y_patch = self.transform(x_patch, y_patch)

                x_patches[sample_n, :, :, :] = x_patch
                y_patches[sample_n, :, :, :] = y_patch
                sample_n += 1

        # ------------------------------------------------------------------ #
        # Phase 2 — compensatory sampling to cover class deficits.            #
        # ------------------------------------------------------------------ #
        if _two_phase and sample_n < self.options['batch_size']:
            n_classes = self.options['n_classes'][self._target_chart]
            target_ratio = np.array(self.options.get(
                'target_class_ratio', [1.0 / n_classes] * n_classes), dtype=float)

            # Count pixel distribution across Phase 1 patches.
            phase1_labels = y_patches[:base_n, self._target_chart_idx].numpy().flatten()
            valid_pixels = phase1_labels[phase1_labels != 255]
            if len(valid_pixels) > 0:
                actual_counts = np.array(
                    [float((valid_pixels == c).sum()) for c in range(n_classes)])
                actual_ratio = actual_counts / actual_counts.sum()
            else:
                actual_ratio = np.ones(n_classes) / n_classes

            # Deficit = how much each class is below its target share.
            deficit = np.maximum(0.0, target_ratio - actual_ratio)

            n_supplement = self.options['batch_size'] - base_n
            if deficit.sum() > 0:
                weights = deficit / deficit.sum()
                supplement_assignments = np.random.choice(
                    n_classes, size=n_supplement, p=weights)
            else:
                supplement_assignments = np.random.randint(n_classes, size=n_supplement)

            max_retries = self.options.get('phase2_max_retries', 20)
            for target_cls in supplement_assignments:
                success = False
                for _ in range(max_retries):
                    bank = self.class_location_bank
                    if int(target_cls) in bank and bank[int(target_cls)]:
                        idx_b = np.random.randint(len(bank[int(target_cls)]))
                        s_id, row, col = bank[int(target_cls)][idx_b]
                        x_patch, y_patch = self._crop_at(s_id, row, col)
                        scene_id = s_id
                    else:
                        x_patch, y_patch = None, None

                    if x_patch is None:
                        # Fall back to random crop if bank entry is invalid.
                        scene_id = np.random.randint(low=0, high=len(self.files))
                        try:
                            if self.downsample:
                                x_patch, y_patch = self.random_crop_downsample(scene_id)
                            else:
                                scene = xr.open_dataset(os.path.join(
                                    self.options['path_to_train_data'],
                                    self.files[scene_id]),
                                    engine='h5netcdf', mask_and_scale=False)
                                x_patch, y_patch = self.random_crop(scene)
                                scene.close()
                        except Exception:
                            continue

                    if x_patch is not None:
                        x_patch = self._augment_x(x_patch, scene_id)
                        if self.do_transform:
                            x_patch, y_patch = self.transform(x_patch, y_patch)
                        x_patches[sample_n, :, :, :] = x_patch
                        y_patches[sample_n, :, :, :] = y_patch
                        sample_n += 1
                        success = True
                        break

                if not success:
                    # Max retries exceeded; skip slot (leave as zeros).
                    sample_n += 1

        if self.do_transform and torch.rand(1) < self.options['data_augmentations']['Cutmix_prob']:
            lam = np.random.beta(self.options['data_augmentations']['Cutmix_beta'],
                                  self.options['data_augmentations']['Cutmix_beta'])
            rand_index = torch.randperm(x_patches.size(0))
            bbx1, bby1, bbx2, bby2 = rand_bbox(x_patches.size(), lam)
            x_patches[:, :, bbx1:bbx2, bby1:bby2] = x_patches[rand_index, :, bbx1:bbx2, bby1:bby2]
            y_patches[:, :, bbx1:bbx2, bby1:bby2] = y_patches[rand_index, :, bbx1:bbx2, bby1:bby2]

        x, y = self.prep_dataset(x_patches, y_patches)
        return x, y


    def save_patch_log(self, save_path, epoch):
        """
        Append patch sampling statistics to a CSV file, then clear the log.

        Mode is controlled by train_options['patch_log_mode']:
          'per_epoch' (default): one row per epoch, columns are total pixel counts
                                 summed across ALL patches in that epoch.
          'per_patch'           : one row per patch, columns are pixel counts for
                                 that individual patch (also records the source file).

        NOTE: requires num_workers=0 to function correctly.

        Parameters
        ----------
        save_path : str
            Path to the CSV file (created on first call, appended thereafter).
        epoch : int
            Current epoch number (written into every row).
        """
        mode = self.options.get('patch_log_mode', 'per_epoch')
        prefix = self._target_chart.lower()
        n_classes = self.options['n_classes'][self._target_chart]
        real_classes = list(range(n_classes))

        if mode == 'per_epoch':
            fieldnames = ['epoch'] + [f'{prefix}_{c}' for c in real_classes] + [f'{prefix}_mask']
        else:  # per_patch
            fieldnames = ['epoch', 'file'] + [f'{prefix}_{c}' for c in real_classes] + [f'{prefix}_mask']

        write_header = not osp.exists(save_path)
        with open(save_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            if mode == 'per_epoch':
                # Aggregate pixel counts across all patches in this epoch into one row.
                totals = {c: 0 for c in real_classes}
                totals[255] = 0
                for entry in self.patch_log:
                    for c in real_classes:
                        totals[c] += entry['sod_dist'].get(c, 0)
                    totals[255] += entry['sod_dist'].get(255, 0)
                row = {'epoch': epoch}
                for c in real_classes:
                    row[f'{prefix}_{c}'] = totals[c]
                row[f'{prefix}_mask'] = totals[255]
                writer.writerow(row)
            else:
                # One row per patch.
                for entry in self.patch_log:
                    row = {'epoch': epoch, 'file': entry['file']}
                    for c in real_classes:
                        row[f'{prefix}_{c}'] = entry['sod_dist'].get(c, 0)
                    row[f'{prefix}_mask'] = entry['sod_dist'].get(255, 0)
                    writer.writerow(row)

        self.patch_log.clear()


class AI4ArcticChallengeTestDataset(Dataset):
    """Pytorch dataset for loading full scenes from the ASID ready-to-train challenge dataset for inference."""

    def __init__(self, options, files,files_reference=None, mode='test'):
        self.options = options
        self.files = files
        self.files_reference = files_reference

        # if mode not in ["train_val", "test_val", "test"]:
        if mode not in ["train", "test", "test_no_gt"]:
            raise ValueError("String variable must be one of 'train', 'test', or 'test_no_gt'")
        self.mode = mode

    def __len__(self):
        """
        Provide the number of iterations. Function required by Pytorch dataset.

        Returns
        -------
        Number of scenes per validation.
        """
        return len(self.files)

    def prep_scene(self, scene, scene_y=None, filename=None):
        """
        Load a full scene for inference/validation. For MyDS, all variables are at
        the same 80m resolution, so no upsampling is needed.

        Parameters
        ----------
        scene :
            Xarray dataset for input features.
        scene_y :
            Optional separate dataset for labels (targets). If None, uses scene.

        Returns
        -------
        x :
            4D torch tensor, ready inference data.
        y :
            Dict with 2D numpy arrays for each chart. None if mode is 'test_no_gt'.
        """
        # All MyDS variables are at the same 80m resolution - load all at once.
        x = torch.from_numpy(
            scene[self.options['train_variables']].to_array().values).unsqueeze(0).float()

        # Downscale if needed
        # 验证/测试时可用 val_downsample_scale 单独控制分辨率，避免大场景 GPU OOM
        if self.mode in ('train', 'test', 'test_no_gt') and self.options.get('val_downsample_scale', 1) != 1:
            effective_scale = self.options['val_downsample_scale']
        else:
            effective_scale = self.options['down_sample_scale']

        if effective_scale != 1:
            x = torch.nn.functional.interpolate(
                x, scale_factor=1/effective_scale,
                mode=self.options['loader_downsampling'])

        if self.mode != 'test_no_gt':
            scene_for_y = scene_y if scene_y is not None else scene
            y_charts = torch.from_numpy(
                scene_for_y[self.options['charts']].isel().to_array().values).unsqueeze(0)
            y_charts = torch.nn.functional.interpolate(
                y_charts, scale_factor=1/effective_scale, mode='nearest')

            y = {}
            for idx, chart in enumerate(self.options['charts']):
                y[chart] = y_charts[:, idx].squeeze().numpy()

            target_chart = self.options.get('target_chart', 'SOD')

            # SOD-specific: remap labels: merge old class 2 (young ice) into class 1
            # (new/young ice). All old labels >=2 (excluding mask=255) shift down by 1.
            if target_chart == 'SOD' and 'SOD' in y:
                sod = y['SOD'].copy()
                valid = sod != 255
                sod[valid & (sod >= 2)] -= 1
                y['SOD'] = sod

            # FLOE-specific label remapping:
            #   1(冰块), 2(小浮冰), 3(中浮冰), 6(冰山) → 255 (忽略)
            #   4(大浮冰) → 1, 5(巨浮冰) → 2
            elif target_chart == 'FLOE' and 'FLOE' in y:
                floe = y['FLOE'].copy()
                floe[(floe >= 1) & (floe <= 3)] = 255
                floe[floe == 6] = 255
                floe[floe == 4] = 1
                floe[floe == 5] = 2
                y['FLOE'] = floe

            # Boundary erosion on validation labels: mask out class-boundary pixels that
            # are most likely mislabeled due to spatial misalignment between CIS ice
            # charts and SAR. Controlled by 'boundary_erosion_iters' (default 0 = off).
            _erode_iters = self.options.get('boundary_erosion_iters', 0)
            if _erode_iters > 0 and target_chart in y:
                y[target_chart] = _erode_sod_boundaries(
                    y[target_chart], erosion_iters=_erode_iters)
        else:
            y = None

        # Append sin/cos month encoding as 2 constant spatial channels.
        if self.options.get('month_encoding', False) and filename is not None:
            s, c = _month_sin_cos(filename)
            H, W = x.shape[-2], x.shape[-1]
            month_ch = torch.tensor([s, c], dtype=torch.float).view(1, 2, 1, 1).expand(1, 2, H, W)
            x = torch.cat([x, month_ch], dim=1)

        # Append HH-HV polarization ratio channel (dB difference).
        # nersc_sar_primary (HH) is channel 0, nersc_sar_secondary (HV) is channel 1.
        if self.options.get('pol_ratio_channel', False):
            pol_ratio = x[:, 0:1, :, :] - x[:, 1:2, :, :]
            x = torch.cat([x, pol_ratio], dim=1)

        return x.float(), y

    def __getitem__(self, idx):
        """
        Get scene. Function required by Pytorch dataset.

        Returns
        -------
        x :
            4D torch tensor; ready inference data.
        y :
            Dict with 2D numpy arrays for each chart. None if mode is 'test_no_gt'.
        cfv_masks :
            Dict of boolean masks for class fill values (label mask). None if 'test_no_gt'.
        tfv_mask :
            2D boolean mask for train fill value (input invalid region / black border).
        name : str
            Filename of scene.
        original_size : tuple
            (H, W) of the scene before any downsampling.
        """
        scene_y = None
        if self.mode == 'test':
            # For MyDS, labels are embedded in the same file as features.
            scene = xr.open_dataset(os.path.join(
                self.options['path_to_test_data'], self.files[idx]), engine='h5netcdf',mask_and_scale=False)
            scene_y = scene
        elif self.mode == 'test_no_gt':
            scene = xr.open_dataset(os.path.join(
                self.options['path_to_test_data'], self.files[idx]), engine='h5netcdf', mask_and_scale=False)
        elif self.mode == 'train':
            scene = xr.open_dataset(os.path.join(
                self.options['path_to_train_data'], self.files[idx]), engine='h5netcdf', mask_and_scale=False)

        x, y = self.prep_scene(scene, scene_y, filename=self.files[idx])
        name = self.files[idx]

        if self.mode != 'test_no_gt':
            cfv_masks = {}
            for chart in self.options['charts']:
                cfv_masks[chart] = (
                    y[chart] == self.options['class_fill_values'][chart]).squeeze()
        else:
            cfv_masks = None

        # Use global_valid_mask as the train fill value mask (more accurate than checking SAR=0).
        if 'global_valid_mask' in self.options['train_variables']:
            mask_idx = list(self.options['train_variables']).index('global_valid_mask')
            tfv_mask = (x.squeeze()[mask_idx, :, :] == 0).squeeze()
        else:
            tfv_mask = (x.squeeze()[0, :, :] == self.options['train_fill_value']).squeeze()

        original_size = scene['nersc_sar_primary'].values.shape

        return x, y, cfv_masks, tfv_mask, name, original_size


def get_variable_options(train_options: dict):
    """
    Set up variable category lists for MyDS dataset.

    In MyDS all input variables share the same 80m pixel spacing as the SAR imagery,
    so no upsampling is required. All train_variables are treated as full-resolution.

    Parameters
    ----------
    train_options: dict
        Dictionary with training options.

    Returns
    -------
    train_options: dict
        Updated with variable category lists.
    """
    # Derive active charts and task weights from target_chart.
    # Set 'target_chart' in config to 'SOD' or 'FLOE' to switch the detection target.
    target_chart = train_options.get('target_chart', 'SOD')
    train_options['charts'] = [target_chart]
    train_options['task_weights'] = [1]

    # Update chart_metric weights: 1 for target, 0 for others.
    for chart in train_options.get('chart_metric', {}):
        train_options['chart_metric'][chart]['weight'] = 1 if chart == target_chart else 0

    # All train variables in MyDS are full-resolution (80m) - no separate AMSR/env grids.
    train_options['sar_variables'] = list(train_options['train_variables'])
    train_options['full_variables'] = list(train_options['charts']) + train_options['sar_variables']
    train_options['amsrenv_variables'] = []
    train_options['auxiliary_variables'] = []

    return train_options
