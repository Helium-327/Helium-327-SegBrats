# -*- coding: UTF-8 -*-
'''

代码说明:    训练流程

Created on      2024/07/23 15:28:23
Author:         @Mr_Robot
State:          loss 可以正常下降，需要进行数据增强
TODO:          1. 添加早停策略
'''

import os
import time
import torch
import pandas as pd
import argparse
from tabulate import tabulate

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.amp import GradScaler

from readDatasets.BraTS import BraTS21_3d
from transforms import data_transform, Compose, RandomCrop3D, Normalize, tioRandomNoise3d, tioRandomGamma3d, tioRandomFlip3d
from utils.writinglog import writeTraininglog
from utils.splitDataList import DataSpliter
from utils.utils_ckpt import save_checkpoint, load_checkpoint
from nets.unet3ds import *
from metrics import EvaluationMetrics

from loss_function import DiceLoss, CELoss, FocalLoss
from train_and_val import train_one_epoch, val_one_epoch

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# constant
RANDOM_SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



now = time.localtime()
detailed_time_str = time.strftime("%Y-%m-%d-%H-%M-%S", now)
log_name = f"training_log_{detailed_time_str}.txt"

def train(model, Metrics, train_loader, val_loader, scaler, optimizer, scheduler, loss_function, num_epochs, device, ckpt_root, results_path, start_epoch, best_val_loss, tb=False,  interval=10):
    """
    模型训练流程
    :param model: 模型
    :param train_loader: 训练数据集
    :param val_loader: 验证数据集
    :param optimizer: 优化器
    :param loss_function: 又称 criterion 损失函数
    :param num_epochs: 训练轮数
    :param device: 设备
    """
    best_epoch = 0

    writer = SummaryWriter(f'runs/{detailed_time_str}_unet_braTS21_3d')
    if not os.path.exists(ckpt_root):
        os.makedirs(ckpt_root)
        
    val_metrics = []
    end_epoch = start_epoch + num_epochs
    
    for epoch in range(start_epoch, end_epoch):
        
        # =============================== 训练过程 ===============================
        print(f"===== Training on [Epoch {epoch+1}/{end_epoch}]:")
        
        train_mean_loss = 0.0
        start_time = time.time()
        # 训练
        train_running_loss, train_et_loss, train_tc_loss, train_wt_loss = train_one_epoch(model, train_loader, scaler, optimizer, loss_function, device)
        
        # 计算平均loss
        train_mean_loss =  train_running_loss / len(train_loader) 
        mean_train_et_loss = train_et_loss / len(train_loader)
        mean_train_tc_loss = train_tc_loss / len(train_loader)
        mean_train_wt_loss = train_wt_loss / len(train_loader)
        
        # scheduler.step(train_mean_loss)
        writer.add_scalars('train/DiceLoss',
                           {'Mean':train_mean_loss, 
                            'ET': mean_train_et_loss, 
                            'TC': mean_train_tc_loss, 
                            'WT': mean_train_wt_loss}, epoch)
        end_time = time.time()
        train_cost_time = end_time - start_time
        # print info
        print(f"@Train mean loss: {train_mean_loss:.4f} @ET loss: {mean_train_et_loss:.4f}; @TC loss: {mean_train_tc_loss:.4f}; @WT loss: {mean_train_wt_loss:.4f}\n"
              f"Cost time: {train_cost_time/60:.2f}mins")
        
        # =============================== 验证过程 ===============================
        if (epoch+1) % interval == 0:
            print(f"===== Validating on [Epoch {epoch+1}/{end_epoch}]:")
            
            # 开始计时
            start_time = time.time()
            
            # 验证
            val_running_loss, val_et_loss, val_tc_loss, val_wt_loss, Metrics_list= val_one_epoch(model, Metrics, val_loader, loss_function, epoch, device)
            
            # 计算平均loss
            val_mean_loss = val_running_loss / len(val_loader)
            mean_val_et_loss = val_et_loss / len(val_loader)
            mean_val_tc_loss = val_tc_loss / len(val_loader)
            mean_val_wt_loss = val_wt_loss / len(val_loader)
            scheduler.step(val_mean_loss)
            
            
            val_scores = {}
            # 记录验证结果
            val_scores['epoch'] = epoch
            val_scores['Dice_scores'] = Metrics_list[0] 
            val_scores['Jaccard_scores'] = Metrics_list[1]
            val_scores['Accuracy_scores'] = Metrics_list[2]
            val_scores['Precision_scores'] = Metrics_list[3]
            val_scores['Recall_scores'] = Metrics_list[4]
            val_scores['F1_scores'] = Metrics_list[5]
            val_scores['F2_scores'] = Metrics_list[6]
            # val_metrics.append(val_scores)
            
            # 记录训练结果
            # tensorboard记录验证结果
            if tb: 
                writer.add_scalars('val/DiceLoss', 
                                {'Mean':val_mean_loss, 
                                    'ET': mean_val_et_loss, 
                                    'TC': mean_val_tc_loss, 
                                    'WT': mean_val_wt_loss},
                                epoch)

                writer.add_scalars('val/Dice_coeff',
                                {'Mean':val_scores['Dice_scores'][0],
                                    'ET': val_scores['Dice_scores'][1],
                                    'TC': val_scores['Dice_scores'][2],
                                    'WT': val_scores['Dice_scores'][3]},
                                epoch)

                writer.add_scalars('val/Jaccard_index',
                                {'Mean':val_scores['Jaccard_scores'][0],
                                    'ET': val_scores['Jaccard_scores'][1],
                                    'TC': val_scores['Jaccard_scores'][2],
                                    'WT': val_scores['Jaccard_scores'][3]},
                                epoch)   

                writer.add_scalars('val/Accuracy',
                                {'Mean':val_scores['Accuracy_scores'][0],
                                    'ET': val_scores['Accuracy_scores'][1],
                                    'TC': val_scores['Accuracy_scores'][2],
                                    'WT': val_scores['Accuracy_scores'][3]},
                                epoch)
                
                writer.add_scalars('val/Precision', 
                                {'Mean':val_scores['Precision_scores'][0],
                                    'ET': val_scores['Precision_scores'][1], 
                                    'TC': val_scores['Precision_scores'][2], 
                                    'WT': val_scores['Precision_scores'][3]},
                                epoch)
                
                writer.add_scalars('val/Recall', 
                                {'Mean':val_scores['Recall_scores'][0], 
                                    'ET': val_scores['Recall_scores'][1], 
                                    'TC': val_scores['Recall_scores'][2], 
                                    'WT': val_scores['Recall_scores'][3]},
                                epoch)
                writer.add_scalars('val/F1', 
                                {'Mean':val_scores['F1_scores'][0], 
                                    'ET': val_scores['F1_scores'][1], 
                                    'TC': val_scores['F1_scores'][2], 
                                    'WT': val_scores['F1_scores'][3]},
                                epoch) 
                writer.add_scalars('val/F2', 
                                {'Mean':val_scores['F2_scores'][0], 
                                    'ET': val_scores['F2_scores'][1], 
                                    'TC': val_scores['F2_scores'][2], 
                                    'WT': val_scores['F2_scores'][3]},
                                epoch)                               
            
            end_time = time.time()
            val_cost_time = end_time - start_time


            # FIXME: 打印指标
            metric_table_header = ["Metric_Name", "MEAN", "ET", "TC", "WT"]
            metric_table_left = ["Dice", "Jaccard", "Accuracy", "Precision", "Recall", "F1", "F2"]
            epoch_string = f"Epoch {epoch} Metrics: \n"
            end_string = f"Cost time: {val_cost_time/60:.2f}mins"

            # 优化点：直接通过映射获取指标名称，避免重复字符串格式化
            def format_value(value, decimals=4):
                # 返回一个格式化后的字符串，保留指定的小数位数
                return f"{value:.{decimals}f}"
            
            metric_scores_mapping = {metric: val_scores[f"{metric}_scores"] for metric in metric_table_left}
            metric_table = [[metric,
                            format_value(metric_scores_mapping[metric][0]),
                            format_value(metric_scores_mapping[metric][1]),
                            format_value(metric_scores_mapping[metric][2]),
                            format_value(metric_scores_mapping[metric][3])] for metric in metric_table_left]
            loss_str = f"Mean Loss: {val_mean_loss:.4f}, ET: {mean_val_et_loss:.4f}, TC: {mean_val_tc_loss:.4f}, WT: {mean_val_wt_loss:.4f}\n"
            table_str = tabulate(metric_table, headers=metric_table_header, tablefmt='grid')
            metrics_info = epoch_string + table_str + '\n' + loss_str + end_string    
            
            # 将指标表格写入日志文件
            writeTraininglog(results_path, metrics_info, log_name)
            print(metrics_info)


            # df = pd.DataFrame(val_metrics)
            # df.to_csv(log_path, index=False)
            
            # ============================= 保存模型 ==============================

        
            if val_mean_loss < best_val_loss:
                best_val_loss = val_mean_loss
                best_epoch = epoch
                
                with open(os.path.join(results_path, "training_logs.txt"), 'a') as f:
                    f.write(f"SCORES ON EPOCH:--{best_epoch}-- ,{detailed_time_str}:\n"
                            f"mean dice : {val_scores['Dice_scores'][0]:.4f};   ET : {val_scores['Dice_scores'][1]:.4f};   TC : {val_scores['Dice_scores'][2]:.4f};  WT : {val_scores['Dice_scores'][3]:.4f}\n")
                checkpoint_path = os.path.join(ckpt_root, f"BraTS21_3d_{detailed_time_str}_{best_epoch}_{val_scores['Dice_scores'][0]:.4f}.pth")
                
                if best_val_loss < 0.5: # 损失小于0.5时保存模型
                    save_checkpoint(model, optimizer, scaler, best_epoch, best_val_loss, checkpoint_path)
                
    print(f"🎉🎉🎉🎉 Train finished. Best val loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    # 训练完成后关闭 SummaryWriter
    writer.close() 
    # df = pd.DataFrame(val_metrics)
    # df.to_csv(os.path.join("./results", "scores.csv"), index=False)
    # 可视化
    
    
def main(args):

    """参数列表格式化输出并保存"""
    # 将参数转换成字典,并输出参数列表
    params = vars(args) 
    params_dict = {}
    params_dict['Parameter']=[str(p[0]) for p in list(params.items())]
    params_dict['Value']=[str(p[1]) for p in list(params.items())]
    params_header = ["Parameter", "Value"]

    # 标准输出
    print(tabulate(params_dict, headers=params_header, tablefmt="grid"))
    # 重定向输出
    writeTraininglog(args.results_path, '='*40 + '\n' + "训练参数" +'\n' + '='*40, log_name)
    writeTraininglog(args.results_path, tabulate(params_dict, headers=params_header, tablefmt="grid"), log_name)


    """创建模型实例"""
    start_epoch = 0
    best_val_loss = float('inf')
    
    assert args.model in ['UNet_3d_22M_32', 'UNet_3d_22M_64', 'UNet_3d_48M', 'UNet_3d_90M', 'UNet_3d_ln', 'UNet_3d_ln2'], "Invalid model name"
    
    if args.model == 'UNet_3d_22M_64':
        model = UNet_3d_22M_64(4, 4)
    elif args.model == 'UNet_3d_48M':
        model = UNet_3d_48M(4, 4)
    elif args.model == 'UNet_3d_90M':
        model = UNet_3d_90M(4, 4)
    elif args.model == 'UNet_3d_ln':
        model = UNet_3d_ln(4, 4)
    elif args.model == 'UNet_3d_ln2':
        model = UNet_3d_ln2(4, 4)
    else:
        model = UNet_3d_22M_32(4, 4)
    
    init_weights_light(model)
    model.to(DEVICE)

    # FIXME: 实现断点训练
    if args.resume:
        print(f"Resuming training from checkpoint {args.resume}")
        checkpoint = torch.load(args.resume)
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint {args.resume}")
        print(f"Best val loss: {best_val_loss:.4f} at epoch {start_epoch}")


    scaler = GradScaler() # 混合精度训练
    MetricsGo = EvaluationMetrics() # 实例化评估指标类
    
    # ====================== 划分数据集 ====================================
    train_csv = os.path.join(args.data_root, "train.csv")
    val_csv = os.path.join(args.data_root, "val.csv")
    test_csv = os.path.join(args.data_root, "test.csv")

    root = args.data_root
    path_data = os.path.join(root, "BraTS2021_Training_Data")
    
    assert os.path.exists(root), f"{root} not exists."
    
    if args.data_split:
        dataspliter =  DataSpliter(path_data, train_split=args.ts, val_split=args.vs, seed=RANDOM_SEED)

        train_list, test_list, val_list = dataspliter.data_split()
        dataspliter.save_as_csv(train_list, train_csv)
        dataspliter.save_as_csv(test_list, val_csv)
        dataspliter.save_as_csv(val_list, test_csv)
    

    # ====================== 载入数据集 ====================================
    TransMethods_train = data_transform(transform=Compose([RandomCrop3D(size=args.trainCropSize),    # 随机裁剪
                                                        # tioRandonCrop3d(size=CropSize),
                                                        tioRandomFlip3d(),                 # 随机翻转
                                                        # tioRandomElasticDeformation3d(),
                                                        # tioZNormalization(),               # 归一化
                                                        tioRandomNoise3d(),
                                                        tioRandomGamma3d(),    
                                                        Normalize(mean=(0.114, 0.090, 0.170, 0.096), std=(0.199, 0.151, 0.282, 0.174)),   # 标准化
                                                        # tioRandomAffine(),          # 随机旋转
                                      ]))
    
    TransMethods_val = data_transform(transform=Compose([RandomCrop3D(size=args.valCropSize),    # 随机裁剪
                                                         Normalize(mean=(0.114, 0.090, 0.170, 0.096), std=(0.199, 0.151, 0.282, 0.174)),   # 标准化
                                                         tioRandomFlip3d(),   
                                      ]))
    """加载数据集"""
    if args.data_scale == 'local':
        assert args.trainset_len and args.valset_len ,"local Training need to set (trainset_len) and (valset_len)!"
        # 载入部分数据集
        train_dataset = BraTS21_3d(train_csv, 
                                   transform=TransMethods_train,
                                   local_train=True, 
                                   length=args.trainset_len)
        
        val_dataset   = BraTS21_3d(val_csv, 
                                   transform=TransMethods_val, 
                                   local_train=True, 
                                   length=args.valset_len)

        # test_dataset  = BraTS21_3d(test_csv, 
        #                            transform=TransMethods_val, 
        #                            local_train=True, 
        #                            length=args.local_val_length)
    
    elif args.data_scale == 'full':        # 载入全部数据集
        ## 全量数据集
        train_dataset = BraTS21_3d(train_csv, 
                                transform=TransMethods_train)
        
        val_dataset   = BraTS21_3d(val_csv, 
                                transform=TransMethods_val)

        # test_dataset  = BraTS21_3d(test_csv, 
        #                         transform=TransMethods_val)
    else:
        raise ValueError("data_scale must be 'local' or 'full'.")
    


    train_loader = DataLoader(train_dataset, 
                              batch_size=args.bs, 
                              num_workers=args.nw,
                              shuffle=True)
    
    val_loader   = DataLoader(val_dataset, 
                              batch_size=args.bs, 
                              num_workers=args.nw,
                              shuffle=False)
    
    # test_loader  = DataLoader(test_dataset, 
    #                           batch_size=args.bs, 
    #                           num_workers=args.nw,
    #                           shuffle=False)

    # ======================= 训练组件 ========================================
    # 优化器
    assert args.optimizer in ['AdamW', 'SGD', 'RMSprop'], \
        f"optimizer must be 'AdamW', 'SGD' or 'RMSprop', but got {args.optimizer}."
    if args.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd) # 会出现梯度爆炸或消失
    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=args.lr, alpha=0.9, eps=1e-8)
    else:
        raise ValueError("optimizer must be 'AdamW', 'SGD' or 'RMSprop'.")
    

    # 调度器
    assert args.scheduler in ['ReduceLROnPlateau', 'CosineAnnealingLR'], \
        f"scheduler must be 'ReduceLROnPlateau' or 'CosineAnnealingLR', but got {args.scheduler}."
    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)
    
    # 损失函数
    assert args.loss in ['DiceLoss', 'CELoss', 'FocalLoss'], \
        f"loss must be 'DiceLoss' or 'CELoss' or 'FocalLoss', but got {args.loss}."
    if args.loss == 'CELoss':
        loss_function = CELoss()
    elif args.loss == 'FocalLoss':
        loss_function = FocalLoss()
    else:
        loss_function = DiceLoss()
    
    # # 早停
    # if args.stop_patience:
    #     earlyStopping = EarlyStopping(model, patience=args.stop_patience)
    
    train(model, 
          Metrics=MetricsGo, 
          train_loader=train_loader,
          val_loader=val_loader, 
          scaler=scaler, 
          optimizer=optimizer,
          scheduler=scheduler,
          loss_function=loss_function,
          num_epochs=args.epochs, 
          device=DEVICE, 
          ckpt_root=args.ckpt_path,
          results_path=args.results_path,
          start_epoch=start_epoch,
          best_val_loss=best_val_loss,
          tb=args.tb,
          interval=args.interval)
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train args")

    parser.add_argument("--data_root" , type=str, default="./brats21_local", help="data root")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints", help="checkpoint root")
    parser.add_argument("--resume", type=str, default=None, help="resume training from checkpoint")
    parser.add_argument("--results_path", type=str, default="./results", help="result path")
    
    parser.add_argument("--model", type=str, default="UNet_3d_22M_32", help="model")
    parser.add_argument("--epochs", type=int, default=20, help="num_epochs")
    parser.add_argument("--nw", type=int, default=8, help="num_workers")
    parser.add_argument("--bs", type=int, default=2, help="batch_size")
    
    parser.add_argument("--input_channels", type=int, default=4, help="input channels")
    parser.add_argument("--output_channels", type=int, default=4, help="output channels")
    parser.add_argument("--trainCropSize", type=lambda x: tuple(map(int, x.split(','))), default=(128, 128, 128), help="crop size")
    parser.add_argument("--valCropSize", type=lambda x: tuple(map(int, x.split(','))), default=(128, 128, 128), help="crop size")
    
    parser.add_argument("--loss", type=str, default="CELoss", help="loss function")
    parser.add_argument("--loss_type", type=str, default="custom", help="loss type to grad")
    
    parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
    
    parser.add_argument("--scheduler", type=str, default="ReduceLROnPlateau", help="scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=3, help="scheduler patience")
    parser.add_argument("--scheduler_factor", type=float, default=0.9, help="scheduler factor")
    
    parser.add_argument("--data_scale", type=str, default="local", help="loading data scale")
    parser.add_argument("--trainset_len", type=int, default=100, help="train length")
    parser.add_argument("--valset_len", type=int, default=12, help="val length")
    parser.add_argument("--interval", type=int, default=1, help="checkpoint interval")
    
    parser.add_argument("--tb", type=bool, default=False, help="Tensorboard True or False")
    parser.add_argument("--data_split", type=bool, default=False, help="data split True or False")
    parser.add_argument("--ts", type=float, default=0.8, help="train_split_rata")
    parser.add_argument("--vs", type=float, default=0.1, help="val_split_rate")
    # parser.add_argument("--stop_patience", type=int, default=10, help="early stopping")
    
    
    args = parser.parse_args()
    main(args=args)


