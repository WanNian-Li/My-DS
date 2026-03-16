#!/usr/bin/env python
# -*-coding:utf-8 -*-

_base_ = ['../_base_/base.py']

train_options = {
    'path_to_train_data': '/root/autodl-tmp/data_myds/',
    'path_to_test_data': '/root/autodl-tmp/data_myds/',
    'compute_classwise_f1score': True,

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

    'batch_size': 16,
    'num_workers': 12,  # Number of parallel processes to fetch data.
    'num_workers_val': 4,  # Number of parallel processes during validation.
    'patch_size': 256,
    'down_sample_scale': 10,
    'unet_conv_filters': [32, 32, 64, 64],
}
