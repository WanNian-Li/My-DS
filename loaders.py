
# -- Built-in modules -- #
import os
import os.path as osp
import csv
from tqdm import tqdm

# -- Third-party modules -- #
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# -- Proprietary modules -- #
from functions import rand_bbox

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
        self._sod_chart_idx = self.options['charts'].index('SOD')

    def __len__(self):
        return self.options['epoch_len']

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
            low=0, high=scene['SOD'].values.shape[0]
            - self.options['patch_size'])
        col_rand = np.random.randint(
            low=0, high=scene['SOD'].values.shape[1]
            - self.options['patch_size'])

        # Discard patches with too many black border (georrectification artifact) pixels.
        if 'global_valid_mask' in self.options['train_variables']:
            valid_mask_patch = scene['global_valid_mask'].isel(
                y=slice(row_rand, row_rand + self.options['patch_size']),
                x=slice(col_rand, col_rand + self.options['patch_size'])).values
            valid_ratio = float(np.mean(valid_mask_patch))
            if valid_ratio < self.options.get('valid_mask_threshold', 0.5):
                return None, None

        sod_patch = scene['SOD'].isel(
            y=slice(row_rand, row_rand + self.options['patch_size']),
            x=slice(col_rand, col_rand + self.options['patch_size'])).values
        if np.sum(sod_patch != self.options['class_fill_values']['SOD']) > 1:

            # Crop all full-resolution variables (MyDS: all variables at same 80m resolution).
            patch[0:len(self.options['full_variables']), :, :] = \
                scene[self.options['full_variables']].isel(
                y=range(row_rand, row_rand + self.options['patch_size']),
                x=range(col_rand, col_rand + self.options['patch_size'])).to_array().values

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
        if np.sum(self.scenes[idx][0, row_rand: row_rand + self.options['patch_size'],
                                   col_rand: col_rand + self.options['patch_size']].numpy()
                  != self.options['class_fill_values']['SOD']) > 1:

            patch[0:len(self.options['full_variables']), :, :] = \
                self.scenes[idx][:, row_rand:row_rand + int(self.options['patch_size']),
                                 col_rand:col_rand + int(self.options['patch_size'])].numpy()

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

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y : Dict
            Dictionary with 3D torch tensors for each chart; reference data for training data x.
        """
        # Placeholder to fill with data.

        x_patches = torch.zeros((self.options['batch_size'], len(self.options['train_variables']),
                                 self.options['patch_size'], self.options['patch_size']))
        y_patches = torch.zeros((self.options['batch_size'], len(self.options['charts']),
                                 self.options['patch_size'], self.options['patch_size']))
        sample_n = 0

        # Continue until batch is full.
        while sample_n < self.options['batch_size']:
            # - Open memory location of scene. Uses 'Lazy Loading'.
            scene_id = np.random.randint(
                low=0, high=len(self.files), size=1).item()

            # - Extract patches
            try:
                if self.downsample:
                    x_patch, y_patch = self.random_crop_downsample(scene_id)
                else:
                    scene = xr.open_dataset(os.path.join(
                        self.options['path_to_train_data'], self.files[scene_id]), engine='h5netcdf', mask_and_scale=False)
                    x_patch, y_patch = self.random_crop(scene)
                    scene.close()
            except FileNotFoundError:
                print(f"File {self.files[scene_id]} not found. Skipping scene.")
                continue

            except Exception as e:
                if self.downsample:
                    print(f"Cropping in {self.files[scene_id]} failed.")
                    print(f"Scene size: {self.scenes[scene_id][0].shape} for crop shape: \
                        ({self.options['patch_size']}, {self.options['patch_size']})")
                    print('Skipping scene.')
                    continue
                else:
                    print(f"Cropping in {self.files[scene_id]} failed.")
                    print(f"Scene size: {scene['SOD'].shape} for crop shape: \
                        ({self.options['patch_size']}, {self.options['patch_size']})")
                    scene.close()
                    print('Skipping scene.')
                    continue

            if x_patch is not None:
                # Compute SOD label flat array once; reused by both filters and patch log.
                sod_flat = y_patch[0, self._sod_chart_idx].numpy().flatten()

                # Filter 1 — SOD label coverage: discard patches where the fraction of
                # invalid SOD pixels (==255, i.e. no label) exceeds sod_invalid_max_ratio.
                # This handles the case where SAR data exists but labels are absent.
                # Config: sod_invalid_max_ratio (float [0,1], default 1.0 = disabled).
                sod_invalid_max = self.options.get('sod_invalid_max_ratio', 1.0)
                if sod_invalid_max < 1.0:
                    invalid_ratio = float((sod_flat == 255).sum()) / len(sod_flat)
                    if invalid_ratio > sod_invalid_max:
                        continue  # discard patch, resample

                # Filter 2 — Water patch rejection: reduce over-sampling of patches that
                # are almost entirely water (SOD=0), computed over valid (non-255) pixels.
                # Config: water_patch_max_ratio (float [0,1], default 1.0 = disabled),
                #         water_rejection_prob  (float [0,1], default 0.0 = disabled).
                water_max_ratio = self.options.get('water_patch_max_ratio', 1.0)
                water_reject_prob = self.options.get('water_rejection_prob', 0.0)
                if water_reject_prob > 0.0 and water_max_ratio < 1.0:
                    valid_mask = sod_flat != 255
                    valid_count = valid_mask.sum()
                    if valid_count > 0:
                        water_ratio = float((sod_flat[valid_mask] == 0).sum()) / valid_count
                        if water_ratio > water_max_ratio and np.random.rand() < water_reject_prob:
                            continue  # discard patch, resample

                if self.do_transform:
                    x_patch, y_patch = self.transform(x_patch, y_patch)

                # -- Stack the scene patches in patches
                x_patches[sample_n, :, :, :] = x_patch
                y_patches[sample_n, :, :, :] = y_patch
                sample_n += 1  # Update the index.

        if self.do_transform and torch.rand(1) < self.options['data_augmentations']['Cutmix_prob']:
            lam = np.random.beta(self.options['data_augmentations']['Cutmix_beta'],
                                  self.options['data_augmentations']['Cutmix_beta'])
            rand_index = torch.randperm(x_patches.size(0))
            bbx1, bby1, bbx2, bby2 = rand_bbox(x_patches.size(), lam)
            x_patches[:, :, bbx1:bbx2, bby1:bby2] = x_patches[rand_index, :, bbx1:bbx2, bby1:bby2]
            y_patches[:, :, bbx1:bbx2, bby1:bby2] = y_patches[rand_index, :, bbx1:bbx2, bby1:bby2]

        # Prepare training arrays

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
        n_sod = self.options['n_classes']['SOD']  # e.g. 5 (classes 0-4)
        real_classes = list(range(n_sod))          # [0,1,2,3,4]

        if mode == 'per_epoch':
            fieldnames = ['epoch'] + [f'sod_{c}' for c in real_classes] + ['sod_mask']
        else:  # per_patch
            fieldnames = ['epoch', 'file'] + [f'sod_{c}' for c in real_classes] + ['sod_mask']

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
                    row[f'sod_{c}'] = totals[c]
                row['sod_mask'] = totals[255]
                writer.writerow(row)
            else:
                # One row per patch.
                for entry in self.patch_log:
                    row = {'epoch': epoch, 'file': entry['file']}
                    for c in real_classes:
                        row[f'sod_{c}'] = entry['sod_dist'].get(c, 0)
                    row['sod_mask'] = entry['sod_dist'].get(255, 0)
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

    def prep_scene(self, scene, scene_y=None):
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
        if self.options['down_sample_scale'] != 1:
            x = torch.nn.functional.interpolate(
                x, scale_factor=1/self.options['down_sample_scale'],
                mode=self.options['loader_downsampling'])

        if self.mode != 'test_no_gt':
            scene_for_y = scene_y if scene_y is not None else scene
            y_charts = torch.from_numpy(
                scene_for_y[self.options['charts']].isel().to_array().values).unsqueeze(0)
            y_charts = torch.nn.functional.interpolate(
                y_charts, scale_factor=1/self.options['down_sample_scale'], mode='nearest')

            y = {}
            for idx, chart in enumerate(self.options['charts']):
                y[chart] = y_charts[:, idx].squeeze().numpy()
        else:
            y = None

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

        x, y = self.prep_scene(scene, scene_y)
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
    # All train variables in MyDS are full-resolution (80m) - no separate AMSR/env grids.
    train_options['sar_variables'] = list(train_options['train_variables'])
    train_options['full_variables'] = list(train_options['charts']) + train_options['sar_variables']
    train_options['amsrenv_variables'] = []
    train_options['auxiliary_variables'] = []

    return train_options
