#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
用法：
python quickstart.py configs/SOD/all.py --wandb-project MyDS --work-dir ./work_dirs/My_DS51 --wandb-name My_DS51

'''

_base_ = ['../_base_/base.py']

train_options = {
    'path_to_train_data': '/root/autodl-tmp/My_dataset/',
    'path_to_test_data': '/root/autodl-tmp/My_dataset/',

    'train_list_path': 'datalists/train_list_small5.json',
    'val_path': 'datalists/val_list5.json',
    'test_path': 'datalists/test_list.json',

    'model_selection': 'swin',

    'compute_classwise_f1score': True,
    'plot_confusion_matrix':True,
    'compile_model': True,  # 开启PyTorch 2.0+ 的图编译加速

    # 'optimizer': {
    #     'type': 'SGD',
    #     'lr': 0.001,  # Optimizer learning rate.
    #     'momentum': 0.9,
    #     'dampening': 0,
    #     'nesterov': False,
    #     'weight_decay': 0.05
    # },
    'optimizer': {
      'type': 'AdamW',
      'lr': 1e-4,
      'b1':0.9,
      'b2': 0.999,
      'eps': 1e-8,
      'weight_decay': 0.05,
    },
    'scheduler': {
        'type': 'CosineAnnealingLR',  # 去掉热重启，使用平滑衰减至 lr_min
        'lr_min': 1e-6,              # 最终学习率下限
    },
    # 'scheduler': {
    #     'type': 'ReduceLROnPlateau',
    #     'factor': 0.5,      # 触发时 lr 减半
    #     'patience': 5,      # 连续 5 个 epoch 无改善触发
    #     'min_lr': 1e-5,     # lr 下限
    #     'threshold': 1e-4,  # 改善量低于此值视为未改善（combined_score 单位：%）
    # },

    'chart_loss': {  # Loss for the task
        'SIC': {
            'type': 'MSELossFromLogits',
            'ignore_index': 255,
        },
        'SOD': {
            'type': 'FocalLoss',
            'ignore_index': 255,
            'gamma': 1.5,
            'weight': [1.0, 1.0, 1.0, 1.5, 1.0],  # per-class alpha，5 classes (0-4)
        },
        'FLOE': {
            'type': 'CrossEntropyLoss',
            'ignore_index': 255,

        },
    },

    'boundary_erosion_iter': 5,
    # 检测目标：'SOD'（海冰发育阶段）或 'FLOE'（海冰形态/浮冰）
    'target_chart': 'SOD',

    'task_weights': [1],

    'early_stop_patience': 30,  # 早停轮次：连续20次验证无改善则停止

    'seed': 10,
    'epochs': 200,
    'epoch_len': 50,  # scale=10时场景预加载入内存，裁剪极快，可采样更多batch

    'num_workers': 18,  # Number of parallel processes to fetch data.
    'num_workers_val': 6,  # Number of parallel processes during validation.
    'prefetch_factor': 4,  # 每个 worker 预取 4 个 batch，保持 GPU 流水线饱满
    'patch_size': 256,
    'batch_size': 64,  # scale=10时patch对应204km范围，预加载路径下可用大batch
    'down_sample_scale': 5,  # 训练降采样10倍：80m→800m，场景预加载入RAM，裁剪速度极快
    'val_freq': 1,   # scale=10时验证场景小（直接整场景推理），可每epoch验证
    'val_downsample_scale': 5,  # 设为1：验证自动沿用down_sample_scale=10，与训练分辨率一致

    'swin_hp': {
        'val_stride': [128, 128],   # 仅Swin模型使用；UNet+scale=10时整场景直接推理，不走滑窗
        'test_stride': [64, 64],
    },
    'unet_conv_filters': [32, 32, 64, 64],
    'unet_dropout': 0.25,  # 添加Dropout2d缓解过拟合（每个DoubleConv块末尾）

    'patch_log_mode': 'per_epoch', # 'per_epoch' 或者 'per_patch'

    # --- Patch 采样过滤 ---
    'sod_invalid_max_ratio': 0.5,   # 无效标签像素 > 50% 时丢弃， 设为 1.0 可关闭此过滤。
    'water_patch_max_ratio': 0.90,   # 水体占有效像素 > 70% 时触发拒绝
    'water_rejection_prob': 0.80,    # 触发后以 90% 的概率丢弃该 patch
    'rare_sampling_classes': [2],  # 新冰/幼冰(1, 合并后) 为稀有类目标
    'rare_sampling_alpha': 0.6,       # 0=均匀采样, 1=完全按稀有类密度采样， 设 rare_sampling_alpha=0.0 或 rare_sampling_classes=[] 可关闭此过滤。
    # HH/HV polarization ratio channel (HH_dB - HV_dB).
    'pol_ratio_channel': True,

    'data_augmentations': {
        'Random_h_flip': 0.5,
        'Random_v_flip': 0.5,
        'Random_rotation_prob': 0.5,
        'Random_rotation': 90,
        'Random_scale_prob': 0.5,
        'Random_scale': (0.9, 1.1),
        'Cutmix_beta': 1.0,
        'Cutmix_prob': 0.5,
    },
}
