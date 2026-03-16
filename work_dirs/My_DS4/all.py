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
        'EpochsPerRestart': 20,  # Number of epochs for the first restart
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
            # Weights: inverse-frequency normalized over classes 0-4 (class 5 absent, class 6 unused)
            # Based on patch_sampling_log: [74.1%, 0.57%, 1.97%, 1.37%, 22.5%, 0%, 0%]
            'weight': [0.022, 2.873, 0.835, 1.196, 0.073, 0.0, 0.0],
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
    'epoch_len': 300,
    'batch_size': 16,

    'num_workers': 0,  # Number of parallel processes to fetch data.
    'num_workers_val': 4,  # Number of parallel processes during validation.
    'patch_size': 256,
    'down_sample_scale': 10,
    'unet_conv_filters': [32, 32, 64, 64],

    'patch_log_mode': 'per_epoch', # 'per_epoch' 或者 'per_patch'

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
