#!/usr/bin/env python
# -*-coding:utf-8 -*-


_base_ = ['../_base_/base.py']

train_options = {
    'path_to_train_data': '/root/autodl-tmp/My_dataset/',
    'path_to_test_data': '/root/autodl-tmp/My_dataset/',

    'train_list_path': 'datalists/train_list_small.json',
    'val_path': 'datalists/val_list.json',
    'test_path': 'datalists/test_list.json',


    'compute_classwise_f1score': True,
    'plot_confusion_matrix':True,
    'compile_model': True,  # 开启PyTorch 2.0+ 的图编译加速

    'optimizer': {
        'type': 'SGD',
        'lr': 0.001,  # Optimizer learning rate.
        'momentum': 0.9,
        'dampening': 0,
        'nesterov': False,
        'weight_decay': 0.01
    },

    'scheduler': {
        'type': 'CosineAnnealingLR',  # 去掉热重启，使用平滑衰减至 lr_min
        'lr_min': 1e-5,              # 最终学习率下限
    },


    'chart_loss': {  # Loss for the task
        'SIC': {
            'type': 'MSELossFromLogits',
            'ignore_index': 255,
        },
        'SOD': {
            'type': 'FocalLoss',
            'ignore_index': 255,
            # 类别权重：根据像素频率倒数设定，水体(0)主动压低，新冰(1)/幼冰(2)大幅提高
            # 频率参考（DS8/DS9实测）: 0=31% 1=3% 2=6% 3=12% 4=39% 5=9%
            'weight': [0.8, 3.0, 2.0, 1.2, 1.0, 1.0],  # 6 classes (0-5)；水体采样已由Filter2控制，loss weight不再额外压低
        },
        'FLOE': {
            'type': 'FocalLoss',
            'ignore_index': 255,
            'weight': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 6 classes (0-5)
        },
    },

    'task_weights': [0, 1, 0],
# 
    'seed': 10,
    'epochs': 70,
    'epoch_len': 100,   # 每epoch采样100个batch，梯度估计更稳定

    'num_workers': 12,  # Number of parallel processes to fetch data.
    'num_workers_val': 4,  # Number of parallel processes during validation.
    'prefetch_factor': 4,  # 每个 worker 预取 4 个 batch，保持 GPU 流水线饱满
    'patch_size': 256,
    'batch_size':64,  # 从16提高至64：down_sample_scale=1时patch仅256×256，大batch才能填满GPU
    'down_sample_scale': 10, # TODO 尝试不同的下采样比例，看看对性能的影响
    'swin_hp': {
        'val_stride': [128, 128],   # UNet scale=1时滑窗步长：128=50%重叠，速度/质量平衡
        'test_stride': [64, 64],
    },
    'unet_conv_filters': [64, 64, 128, 128],
    'unet_dropout': 0.25,  # 添加Dropout2d缓解过拟合（每个DoubleConv块末尾）

    'patch_log_mode': 'per_epoch', # 'per_epoch' 或者 'per_patch'

    # --- Patch 采样过滤 ---
    # Filter 1: SOD 标签覆盖率过滤
    # patch 内 SOD==255（无标签）的像素占全部像素的比例超过此阈值时，丢弃重采样。
    # 解决 SAR 有值但标签缺失的区域被大量采入的问题。
    # 设为 1.0 可关闭此过滤。
    'sod_invalid_max_ratio': 0.5,   # 无效标签像素 > 50% 时丢弃

    # Filter 2: 水体 patch 降采样
    # 有效像素(非255)中水体(SOD=0)占比超过 max_ratio 的 patch，
    # 以 rejection_prob 的概率被丢弃重采样，保留少量水体 patch 维持边界识别能力。
    # 设 water_rejection_prob=0.0 或 water_patch_max_ratio=1.0 可关闭此过滤。
    'water_patch_max_ratio': 0.80,   # 水体占有效像素 > 80% 时触发拒绝
    'water_rejection_prob': 0.85,    # 触发后以 85% 的概率丢弃该 patch

    # --- 稀有类加权采样 ---
    # Filter 3: 根据 patch 内稀有 SOD 类的像素占比，对 patch 进行概率性接受。
    # 接受概率 = (1 - alpha) + alpha * rare_frac
    #   rare_frac=0 时，patch 以 (1-alpha) 的概率被接受（非零保留，保持模型见过非稀有patch）
    #   rare_frac=1 时，patch 始终被接受
    # 设 rare_sampling_alpha=0.0 或 rare_sampling_classes=[] 可关闭此过滤。
    'rare_sampling_classes': [1, 2],  # 新冰(1)、幼冰(2) 为稀有类目标
    'rare_sampling_alpha': 0.3,       # 0=均匀采样, 1=完全按稀有类密度采样

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
