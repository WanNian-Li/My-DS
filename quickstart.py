import argparse
import csv
import json
import random
import os
import os.path as osp
import shutil
from icecream import ic
import pathlib
import warnings
import torchmetrics

warnings.filterwarnings("ignore")

import numpy as np
import torch
from mmcv import Config, mkdir_or_exist
from tqdm import tqdm  # Progress bar

import wandb
# Functions to calculate metrics and show the relevant chart colorbar.
from functions import compute_metrics, save_best_model, load_model, slide_inference, \
    batched_slide_inference, water_edge_metric, class_decider, create_train_validation_and_test_scene_list, \
    get_scheduler, get_optimizer, get_loss, get_model, compute_classwise_IoU

# Load consutme loss function
from losses import WaterConsistencyLoss
# Custom dataloaders for regular training and validation.
from loaders import get_variable_options, AI4ArcticChallengeDataset, AI4ArcticChallengeTestDataset
#  get_variable_options

# -- Built-in modules -- #
from utils import colour_str
from test_upload_function import test


def parse_args():
    parser = argparse.ArgumentParser(description='Train Default U-NET segmentor')

    # Mandatory arguments
    parser.add_argument('config', type=pathlib.Path, help='train config file path',)
    parser.add_argument('--wandb-project', required=True, help='Name of wandb project')
    parser.add_argument('--wandb-name', default='MyDS', help='Name of wandb run (default: MyDS)')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--seed', default=None,
                        help='the seed to use, if not provided, seed from config file will be taken')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume-from', type=pathlib.Path, default=None, # 恢复训练
                       help='Resume Training from checkpoint, it will use the \
                        optimizer and schduler defined on checkpoint')
    group.add_argument('--finetune-from', type=pathlib.Path, default=None,  # 微调/迁移学习
                       help='Start new tranining using the weights from checkpoitn')

    args = parser.parse_args()

    return args


def save_epoch_sod_distribution(save_path, epoch, sod_class_counts, sod_mask_count):
    """Append one row of epoch-level SOD pixel distribution to CSV."""
    fieldnames = ['epoch'] + [f'sod_{c}' for c in range(len(sod_class_counts))] + ['sod_mask']
    write_header = not osp.exists(save_path)

    with open(save_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        row = {'epoch': epoch, 'sod_mask': int(sod_mask_count)}
        for c, count in enumerate(sod_class_counts):
            row[f'sod_{c}'] = int(count)
        writer.writerow(row)


def train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer, scheduler, start_epoch=0):
    '''
    Trains the model.

    '''
    best_combined_score = -np.Inf  # Best weighted model score.

    # Early stopping
    early_stop_patience = train_options.get('early_stop_patience', 10)
    early_stop_counter = 0

    # 验证时使用原始（未编译）模型，避免动态形状触发 recompile 警告
    net_val = net._orig_mod if train_options.get('compile_model') and hasattr(net, '_orig_mod') else net

    # 训练集 IoU（仅对有效任务）
    sod_idx = train_options['charts'].index('SOD')
    compute_train_sod_iou = (train_options['task_weights'][sod_idx] != 0)
    sod_n_classes = train_options['n_classes']['SOD']
    if compute_train_sod_iou:
        train_iou_metric = torchmetrics.classification.MulticlassJaccardIndex(
            num_classes=sod_n_classes, average='none', ignore_index=255).to(device)

    loss_ce_functions = {chart: get_loss(train_options['chart_loss'][chart]['type'], chart=chart, **train_options['chart_loss'][chart]).to(device)
                         for chart in train_options['charts']}

    loss_water_edge_consistency = WaterConsistencyLoss()
    print('Training...')
    # -- Training Loop -- #
    for epoch in tqdm(iterable=range(start_epoch, train_options['epochs'])):
        # gc.collect()  # Collect garbage to free memory.
        train_loss_sum = torch.tensor([0.])  # To sum the training batch losses during the epoch.
        cross_entropy_loss_sum = torch.tensor([0.])  # To sum the training cross entropy batch losses during the epoch.
        # To sum the training edge consistency batch losses during the epoch.
        edge_consistency_loss_sum = torch.tensor([0.])

        val_loss_sum = torch.tensor([0.])  # To sum the validation batch losses during the epoch.
        # To sum the validation cross entropy batch losses during the epoch.
        val_cross_entropy_loss_sum = torch.tensor([0.])
        # To sum the validation cedge consistency batch losses during the epoch.
        val_edge_consistency_loss_sum = torch.tensor([0.])

        # Aggregate SOD label distribution over all training pixels in this epoch.
        sod_num_classes = train_options['n_classes']['SOD']
        sod_epoch_class_counts = np.zeros(sod_num_classes, dtype=np.int64)
        sod_epoch_mask_count = 0

        net.train()  # Set network to evaluation mode.


        #===============================================================#
        #============================训练循环============================#
        #===============================================================#
        for i, (batch_x, batch_y) in enumerate(tqdm(iterable=dataloader_train, total=train_options['epoch_len'],
                                                    colour='red')):
            # torch.cuda.empty_cache()  # Empties the GPU cache freeing up memory.
            train_loss_batch = torch.tensor([0.]).to(device)  # Reset from previous batch.
            edge_consistency_loss = torch.tensor([0.]).to(device)
            cross_entropy_loss = torch.tensor([0.]).to(device)
            # - Transfer to device.
            batch_x = batch_x.to(device, non_blocking=True)

            # Epoch-level SOD pixel stats are computed in the main process, so
            # this works with num_workers > 0.
            if 'SOD' in batch_y:
                sod_batch = batch_y['SOD']
                for c in range(sod_num_classes):
                    sod_epoch_class_counts[c] += int((sod_batch == c).sum().item())
                sod_epoch_mask_count += int((sod_batch == 255).sum().item())

            # - Mixed precision training. (Saving memory)
            with torch.cuda.amp.autocast():

                #=========模型前向传播
                output = net(batch_x)
                # breakpoint()

                #==========计算损失
                for chart, weight in zip(train_options['charts'], train_options['task_weights']):
                    
                    # 如果设置了'edge_consistency_loss'参数，可以计算水体一致性损失
                    if train_options['edge_consistency_loss'] != 0:
                        edge_consistency_loss = loss_water_edge_consistency(output)

                    cross_entropy_loss += weight * loss_ce_functions[chart](
                        output[chart], batch_y[chart].to(device))
            #===========根据是否计算水体一致性损失，得到最终的训练损失
            if train_options['edge_consistency_loss'] != 0:
                a = train_options['edge_consistency_loss']
                edge_consistency_loss = a*loss_water_edge_consistency(output)
                train_loss_batch = cross_entropy_loss + edge_consistency_loss
            else:
                train_loss_batch = cross_entropy_loss

            # - Reset gradients from previous pass.
            optimizer.zero_grad()
            # - Backward pass.
            train_loss_batch.backward()
            # - Optimizer step
            optimizer.step()
            # - Scheduler step（ReduceLROnPlateau 改为 per-epoch，其余类型仍 per-batch）
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            # - Add batch loss.
            train_loss_sum += train_loss_batch.detach().item()
            cross_entropy_loss_sum += cross_entropy_loss.detach().item()
            edge_consistency_loss_sum += edge_consistency_loss.detach().item()

            # 更新训练集 SOD IoU 统计
            if compute_train_sod_iou:
                with torch.no_grad():
                    sod_pred_train = output['SOD'].detach().float().argmax(dim=1)
                    train_iou_metric.update(sod_pred_train, batch_y['SOD'].to(device))

        #===========一个Epoch结束，计算平均损失
        train_loss_epoch = torch.true_divide(train_loss_sum, i + 1).detach().item()
        cross_entropy_epoch = torch.true_divide(cross_entropy_loss_sum, i + 1).detach().item()
        edge_consistency_epoch = torch.true_divide(edge_consistency_loss_sum, i + 1).detach().item()

        # 保存本epoch的SOD像素分布统计（支持 num_workers > 0）
        patch_log_path = osp.join(cfg.work_dir, 'patch_sampling_log.csv')
        save_epoch_sod_distribution(
            patch_log_path,
            epoch,
            sod_epoch_class_counts,
            sod_epoch_mask_count)

        # 计算并打印/记录训练集 SOD IoU
        if compute_train_sod_iou:
            train_iou_per_class = train_iou_metric.compute()  # shape: (n_classes,)
            train_miou = train_iou_per_class.mean()
            train_iou_metric.reset()
            print(f"Train SOD mIoU: {train_miou:.4f}")
            print(f"Train SOD IoU per class: {train_iou_per_class}")
            wandb.log({"Train SOD mIoU": train_miou.item()}, step=epoch)
            for c, iou_c in enumerate(train_iou_per_class):
                wandb.log({f"Train SOD/IoU Class {c}": iou_c.item()}, step=epoch)

        #===============================================================#
        #===========================验证循环=============================#
        #===============================================================#
        val_freq = train_options.get('val_freq', 1)
        do_validate = (epoch % val_freq == 0) or (epoch == train_options['epochs'] - 1)

        if not do_validate:
            # 跳过验证，只记录训练指标
            wandb.log({"Train Epoch Loss": train_loss_epoch,
                       "Train Cross Entropy Epoch Loss": cross_entropy_epoch,
                       "Train Water Consistency Epoch Loss": edge_consistency_epoch,
                       "Learning Rate": optimizer.param_groups[0]["lr"]}, step=epoch)
            print(f"Train Epoch Loss: {train_loss_epoch:.3f}")
            continue

        # - Stores the output and the reference pixels to calculate the scores after inference on all the scenes.
        torch.cuda.empty_cache()  # 释放训练残留的缓存显存，为滑窗推理腾出空间
        outputs_flat = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
        inf_ys_flat = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
        # Outputs mask by train fill values
        outputs_tfv_mask = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
        net.eval()  # Set network to evaluation mode.
        print('Validating...')
        # - Loops though scenes in queue.
        for i, (inf_x, inf_y, cfv_masks, tfv_mask, name, original_size) in enumerate(tqdm(iterable=dataloader_val,
                                                                            total=len(train_options['validate_list']),
                                                                            colour='green')):
            torch.cuda.empty_cache()
            # Reset from previous batch.
            # train fill value mask
            # tfv_mask = (inf_x.squeeze()[0, :, :] == train_options['train_fill_value']).squeeze()
            val_loss_batch = torch.tensor([0.]).to(device)
            val_edge_consistency_loss = torch.tensor([0.]).to(device)
            val_cross_entropy_loss = torch.tensor([0.]).to(device)
            # - Ensures that no gradients are calculated, which otherwise take up a lot of space on the GPU.
            with torch.no_grad(), torch.cuda.amp.autocast():
                inf_x = inf_x.to(device, non_blocking=True)

                #==================根据model_selection选择swin滑动推理还是普通模型的直接推理
                if (train_options['model_selection'] == 'swin' or
                        train_options['down_sample_scale'] == 1):
                    output = slide_inference(inf_x, net_val, train_options, 'val')
                    # output = batched_slide_inference(inf_x, net_val, train_options, 'val')
                else:
                    output = net_val(inf_x)

                for chart, weight in zip(train_options['charts'], train_options['task_weights']):

                    val_cross_entropy_loss += weight * loss_ce_functions[chart](output[chart],
                                                                                inf_y[chart].unsqueeze(0).long().to(device))

                if train_options['edge_consistency_loss'] != 0:
                    a = train_options['edge_consistency_loss']
                    val_edge_consistency_loss = a*loss_water_edge_consistency(output)

            val_loss_batch = val_cross_entropy_loss + val_edge_consistency_loss

            # - Final output layer, and storing of non masked pixels.
            for chart in train_options['charts']:
                output[chart] = class_decider(output[chart], train_options, chart)
                # output[chart] = torch.argmax(
                #     output[chart], dim=1).squeeze()
                outputs_flat[chart] = torch.cat((outputs_flat[chart], output[chart][~cfv_masks[chart]]))
                outputs_tfv_mask[chart] = torch.cat((outputs_tfv_mask[chart], output[chart][~tfv_mask]))
                inf_ys_flat[chart] = torch.cat((inf_ys_flat[chart], inf_y[chart]
                                                [~cfv_masks[chart]].to(device, non_blocking=True)))
            # - Add batch loss.
            val_loss_sum += val_loss_batch.detach().item()
            val_cross_entropy_loss_sum += val_cross_entropy_loss.detach().item()
            val_edge_consistency_loss_sum += val_edge_consistency_loss.detach().item()

        #====================验证结束后，计算综合指标
        val_loss_epoch = torch.true_divide(val_loss_sum, i + 1).detach().item()
        val_cross_entropy_epoch = torch.true_divide(val_cross_entropy_loss_sum, i + 1).detach().item()
        val_edge_consistency_epoch = torch.true_divide(val_edge_consistency_loss_sum, i + 1).detach().item()

        # - Compute the relevant scores.
        print('Computing Metrics on Val dataset')
        combined_score, scores = compute_metrics(true=inf_ys_flat, pred=outputs_flat, charts=train_options['charts'],
                                                 metrics=train_options['chart_metric'], num_classes=train_options['n_classes'])

        water_edge_accuarcy = water_edge_metric(outputs_tfv_mask, train_options)

        if train_options['compute_classwise_f1score']:
            from functions import compute_classwise_f1score, compute_overall_accuracy, compute_mIoU

            # 计算每类的 F1
            classwise_scores = compute_classwise_f1score(true=inf_ys_flat, pred=outputs_flat,
                                                         charts=train_options['charts'], num_classes=train_options['n_classes'])

            # 计算 OA
            oa_scores = compute_overall_accuracy(true=inf_ys_flat, pred=outputs_flat, charts=train_options['charts'])

            # 计算 mIoU 和 per-class IoU
            miou_scores = compute_mIoU(true=inf_ys_flat, pred=outputs_flat,
                                       charts=train_options['charts'], num_classes=train_options['n_classes'])
            classwise_iou_scores = compute_classwise_IoU(true=inf_ys_flat, pred=outputs_flat,
                                                         charts=train_options['charts'], num_classes=train_options['n_classes'])
            
        print("")
        print(f"Epoch {epoch} score:")

        for chart in train_options['charts']:
            # 跳过 task_weight=0 的任务，避免无意义的指标日志
            if train_options['chart_metric'][chart]['weight'] == 0:
                continue

            print(f"{chart} {train_options['chart_metric'][chart]['func'].__name__}: {scores[chart]}%")

            # Log in wandb the SIC r2_metric, SOD f1_metric and FLOE f1_metric
            wandb.log({f"{chart} {train_options['chart_metric'][chart]['func'].__name__}": scores[chart]}, step=epoch)

            #===============计算标签各类别的F1分数
            if train_options['compute_classwise_f1score']:
                for index, class_score in enumerate(classwise_scores[chart]):
                    wandb.log({f"{chart}/Class: {index}": class_score.item()}, step=epoch)
                print(f"{chart} F1 score:", classwise_scores[chart])

                # 打印或记录 OA
                print(f"{chart} OA:", oa_scores[chart])
                wandb.log({f"{chart}/OA": oa_scores[chart]}, step=epoch)

                # 打印/记录 mIoU
                print(f"{chart} mIoU:", miou_scores[chart])
                wandb.log({f"{chart}/mIoU": miou_scores[chart]}, step=epoch)

                # 打印/记录 per-class IoU
                print(f"{chart} IoU per class:", classwise_iou_scores[chart])
                for c, iou_c in enumerate(classwise_iou_scores[chart]):
                    wandb.log({f"{chart}/IoU Class {c}": iou_c.item()}, step=epoch)

        # ReduceLROnPlateau：用验证得分驱动 lr 调整
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_before = optimizer.param_groups[0]['lr']
            scheduler.step(combined_score)
            lr_after = optimizer.param_groups[0]['lr']
            if lr_after < lr_before:
                print(f"ReduceLROnPlateau: lr {lr_before:.2e} → {lr_after:.2e}")

        print(f"Combined score: {combined_score}%")
        print(f"Train Epoch Loss: {train_loss_epoch:.3f}")
        print(f"Train Cross Entropy Epoch Loss: {cross_entropy_epoch:.3f}")
        print(f"Train Water Consistency Epoch Loss: {edge_consistency_epoch:.3f}")
        print(f"Validation Epoch Loss: {val_loss_epoch:.3f}")
        print(f"Validation Cross Entropy Epoch Loss: {val_cross_entropy_epoch:.3f}")
        print(f"Validation val_edge_consistency_loss: {val_edge_consistency_epoch:.3f}")
        print(f"Water edge Accuarcy: {water_edge_accuarcy}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Log combine score and epoch loss to wandb
        wandb.log({"Combined score": combined_score,    # 综合得分（验证集）
                   "Train Epoch Loss": train_loss_epoch,
                   "Train Cross Entropy Epoch Loss": cross_entropy_epoch,
                   "Train Water Consistency Epoch Loss": edge_consistency_epoch,
                   "Validation Epoch Loss": val_loss_epoch,
                   "Validation Cross Entropy Epoch Loss": val_cross_entropy_epoch,
                   "Validation Water Consistency Epoch Loss": val_edge_consistency_epoch,
                   "Water Consistency Accuarcy": water_edge_accuarcy,
                   "Learning Rate": optimizer.param_groups[0]["lr"]}, step=epoch)

        # If the scores is better than the previous epoch, then save the model and rename the image to best_validation.

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            early_stop_counter = 0

            # Log the best combine score, and the metrics that make that best combine score in summary in wandb.
            wandb.run.summary[f"While training/Best Combined Score"] = best_combined_score
            wandb.run.summary[f"While training/Water Consistency Accuarcy"] = water_edge_accuarcy
            for chart in train_options['charts']:
                wandb.run.summary[f"While training/{chart} {train_options['chart_metric'][chart]['func'].__name__}"] = scores[chart]
            wandb.run.summary[f"While training/Train Epoch Loss"] = train_loss_epoch

            # Save the best model in work_dirs
            model_path = save_best_model(cfg, train_options, net, optimizer, scheduler, epoch)

            wandb.save(model_path)
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}. Best score: {best_combined_score:.3f}%")
                break

    del inf_ys_flat, outputs_flat  # Free memory.
    return model_path


def create_dataloaders(train_options):
    '''
    Create train and validation dataloader based on the train and validation list inside train_options.

    '''
    # Custom dataset and dataloader.
    dataset = AI4ArcticChallengeDataset(
        files=train_options['train_list'], options=train_options, do_transform=True)

    prefetch = train_options.get('prefetch_factor', 2) if train_options['num_workers'] > 0 else None
    dataloader_train = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=True, num_workers=train_options['num_workers'],
        pin_memory=True, prefetch_factor=prefetch)
    # - Setup of the validation dataset/dataloader. The same is used for model testing in 'test_upload.ipynb'.

    dataset_val = AI4ArcticChallengeTestDataset(
        options=train_options, files=train_options['validate_list'], mode='train')

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)

    return dataloader_train, dataloader_val
    # return dataloader_val


def main():
    args = parse_args()
    ic(args.config)
    cfg = Config.fromfile(args.config)
    train_options = cfg.train_options
    # Get options for variables, amsrenv grid, cropping and upsampling.
    train_options = get_variable_options(train_options)
    # generate wandb run id, to be used to link the run with test_upload
    id = wandb.util.generate_id()

    #===========设置随机种子
    if train_options['seed'] != -1 and args.seed == None:
        # set seed for everything
        if args.seed != None:
            seed = int(args.seed)
        else:
            seed = train_options['seed']
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = True
        print(f"Seed: {seed}")
    else:
        print("Random Seed Chosen")

    #===========确定工作目录（保存日志和权重文件）
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        if not train_options['cross_val_run']:
            cfg.work_dir = osp.join('./work_dir',
                                    osp.splitext(osp.basename(args.config))[0])
        else:
            # from utils import run_names
            run_name = id
            cfg.work_dir = osp.join('./work_dir',
                                    osp.splitext(osp.basename(args.config))[0], run_name)
    ic(cfg.work_dir)
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    # cfg_path = osp.join(cfg.work_dir, osp.basename(args.config))


    #=========== GPU设置
    # Get GPU resources.
    if torch.cuda.is_available():
        print(colour_str('GPU available!', 'green'))
        print('Total number of available devices: ',
              colour_str(torch.cuda.device_count(), 'orange'))
        
        # Check if NVIDIA V100, A100, or H100 is available for torch compile speed up
        if train_options['compile_model']:
            gpu_ok = False
            major, minor  = torch.cuda.get_device_capability()
            if major >= 7:
                gpu_ok = True
            
            if not gpu_ok:
                warnings.warn(
                    colour_str("GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower than expected.", 'red')
                )

        # Setup device to be used
        device = torch.device(f"cuda:{train_options['gpu_id']}")

    else:
        print(colour_str('GPU not available.', 'red'))
        device = torch.device('cpu')
    print('GPU setup completed!')

    #===========根据"model_selection"选择模型，并设置优化器和学习率调度器
    net = get_model(train_options, device)
    if train_options['compile_model']:
        net = torch.compile(net)
    optimizer = get_optimizer(train_options, net)
    scheduler = get_scheduler(train_options, optimizer)

    #===========恢复训练或微调模型
    if args.resume_from is not None:
        print(f"\033[91m Resuming work from {args.resume_from}\033[0m")
        epoch_start = load_model(net, args.resume_from, optimizer, scheduler)
    elif args.finetune_from is not None:
        print(f"\033[91m Finetune model from {args.finetune_from}\033[0m")
        _ = load_model(net, args.finetune_from)

    #===========Wandb设置
    # if not train_options['cross_val_run']:
    #     wandb.init(name=osp.splitext(osp.basename(args.config))[0], project=args.wandb_project,
    #                entity="liwannian-zhejiang-university", config=train_options)
    # else:
    #     wandb.init(name=osp.splitext(osp.basename(args.config))[0]+'-'+run_name, group=osp.splitext(osp.basename(args.config))[0], project=args.wandb_project,
    #                entity="liwannian-zhejiang-university", config=train_options)
    if not train_options['cross_val_run']:
        wandb.init(name=args.wandb_name, project=args.wandb_project,
                   entity="liwannian-zhejiang-university", config=train_options)
    else:
        wandb.init(name=args.wandb_name, group=osp.splitext(osp.basename(args.config))[0], project=args.wandb_project,
                   entity="liwannian-zhejiang-university", config=train_options)

    # Define the metrics and make them such that they are not added to the summary
    wandb.define_metric("Train Epoch Loss", summary="none")     # 训练集上的总损失
    wandb.define_metric("Train Cross Entropy Epoch Loss", summary="none") # 训练集上的交叉熵损失
    wandb.define_metric("Train Water Consistency Epoch Loss", summary="none") # 训练集上的水体一致性损失
    wandb.define_metric("Validation Epoch Loss", summary="none") # 验证集上的总损失
    wandb.define_metric("Validation Cross Entropy Epoch Loss", summary="none") # 验证集上的交叉熵损失
    wandb.define_metric("Validation Water Consistency Epoch Loss", summary="none") # 验证集上的水体一致性损失
    wandb.define_metric("Combined score", summary="none") # 综合得分
    wandb.define_metric("SIC r2_metric", summary="none") # SIC的R2指标
    wandb.define_metric("SOD f1_metric", summary="none") # SOD的F1指标
    wandb.define_metric("FLOE f1_metric", summary="none") # FLOE的F1指标
    wandb.define_metric("Water Consistency Accuarcy", summary="none") # 水体一致性准确率
    wandb.define_metric("Learning Rate", summary="none") # 学习率
    wandb.save(str(args.config))
    print(colour_str('Save Config File', 'green'))

    

    # ===========创建数据加载器
    
    create_train_validation_and_test_scene_list(train_options)
    dataloader_train, dataloader_val = create_dataloaders(train_options) # dataloader_val 用于训练过程中验证数据

    wandb.config['validate_list'] = train_options['validate_list']
    print('Data setup complete.')

    # ## Example of model training and validation loop
    # A simple model training loop following by a simple validation loop. Validation is carried out on full scenes,
    #  i.e. no cropping or stitching. If there is not enough space on the GPU, then try to do it on the cpu.
    #  This can be done by using 'net = net.cpu()'.


    print('-----------------------------------')
    print('Starting Training')
    print('-----------------------------------')
    if args.resume_from is not None:
        checkpoint_path = train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer,
                                scheduler, epoch_start)
    else:
        checkpoint_path = train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer,
                                scheduler)

    # 加载已训练好的模型
    # checkpoint_path = "./work_dirs/Full_model2/best_model_Full_model2.pth"  
    print('-----------------------------------')
    print('Training Complete')
    print('-----------------------------------')



    print('-----------------------------------')
    print('Staring Validation with best model')
    print('-----------------------------------')

    # this is for valset 1 visualization along with gt
    test('val', net, checkpoint_path, device, cfg.deepcopy(), train_options['validate_list'], 'Cross Validation')


    print('-----------------------------------')
    print('Completed validation')
    print('-----------------------------------')




    print('-----------------------------------')
    print('Starting testing with best model')
    print('-----------------------------------')

    # this is for test path along with gt after the gt has been released
    # test('test', net, checkpoint_path, device, cfg.deepcopy(), train_options['test_list'], 'Test', train_options['test_list_reference'])

    print('-----------------------------------')
    print('Completed testing')
    print('-----------------------------------')


    # finish the wandb run
    wandb.finish()


if __name__ == '__main__':
    main()
