#!/usr/bin/env python
# -*-coding:utf-8 -*-


_base_ = ['../_base_/base.py']

train_options = {
    'path_to_train_data': '/root/autodl-tmp/My_dataset/',
    'path_to_test_data': '/root/autodl-tmp/My_dataset/',

    'train_list_path': 'datalists/train_list.json',
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
        'type': 'CosineAnnealingWarmRestartsLR',  # Name of the schedulers
        'EpochsPerRestart': 10,  # 缩短至10让学习率更快衰减
        # This number will be used to increase or descrase the number of epochs to restart after each restart.
        'RestartMult': 1,
        'lr_min': 0,  # Minimun learning rate
    },


    'chart_loss': {  # Loss for the task
        'SIC': {
            'type': 'MSELossFromLogits',
            'ignore_index': 255,
        },
        'SOD': {
            'type': 'CrossEntropyLoss',
            'ignore_index': 255,
            'weight': [0.022, 2.873, 0.835, 1.196, 0.5],  # 5 classes (0-4)，多年冰(5)已在数据加载时映射为255
        },
        'FLOE': {
            'type': 'CrossEntropyLoss',
            'ignore_index': 255,

        },
    },

    'task_weights': [0, 1, 0],
# 
    'seed': 10,
    'epochs': 70,
    'epoch_len': 100,   # TODO 原本使用300，考虑到目前数据量较低，降低 epoch_len 防止过拟合
    'batch_size': 16,

    'num_workers': 12,  # Number of parallel processes to fetch data.
    'num_workers_val': 4,  # Number of parallel processes during validation.
    'patch_size': 256,
    'down_sample_scale': 5, # TODO 尝试不同的下采样比例，看看对性能的影响
    'unet_conv_filters': [64, 64, 128, 128],

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
    'water_patch_max_ratio': 0.9,   # 水体占有效像素 > 90% 时触发拒绝
    'water_rejection_prob': 0.8,    # 触发后以 80% 的概率丢弃该 patch

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
