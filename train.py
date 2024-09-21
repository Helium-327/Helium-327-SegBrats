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
import readline # 解决input()无法使用Backspace的问题, 不能删掉
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter

from train_and_val import train_one_epoch, val_one_epoch
from utils.log_writer import custom_logger
from utils.ckpt_save_load import save_checkpoint, load_checkpoint
from nets.unet3ds import *
from utils.get_commits import *


os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# constant
RANDOM_SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



date_time_str = get_current_date() + ' ' + get_current_time()
def train(model, Metrics, train_loader, val_loader, scaler, optimizer, scheduler, loss_function, num_epochs, device, results_dir, logs_path, start_epoch, best_val_loss, tb=False,  interval=10, save_loss_threshold=0.4):
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
    end_epoch = start_epoch + num_epochs
    model_name = model.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    scheduler_name = scheduler.__class__.__name__
    loss_func_name = loss_function.__class__.__name__
    tb_dir = os.path.join(results_dir, f'tensorBoard/{model_name}_braTS21_{date_time_str}')
    ckpt_dir = os.path.join(results_dir, f'checkpoints/{model_name}_braTS21_{date_time_str}')
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)
        
    
    for epoch in range(start_epoch, end_epoch):
        
        # =============================== 训练过程 ===============================
        print(f"=== Training on [Epoch {epoch+1}/{end_epoch}] ===:")
        
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
        print(f"- Train mean loss: {train_mean_loss:.4f}\n"
              f"- ET loss: {mean_train_et_loss:.4f}\n"
              f"- TC loss: {mean_train_tc_loss:.4f}\n"
              f"- WT loss: {mean_train_wt_loss:.4f}\n"
              f"- Cost time: {train_cost_time/60:.2f}mins ⏱️\n")
        
        # =============================== 验证过程 ===============================
        if (epoch+1) % interval == 0:
            print(f"=== Validating on [Epoch {epoch+1}/{end_epoch}] ===:")
            
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
            
            
            # 记录验证结果
            val_scores = {}
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


            
            """-------------------------------------- 打印指标 --------------------------------------------------"""
            metric_table_header = ["Metric_Name", "MEAN", "ET", "TC", "WT"]
            metric_table_left = ["Dice", "Jaccard", "Accuracy", "Precision", "Recall", "F1", "F2"]
            val_info_str =f" ===  Epoch {epoch} ===\n"\
                            f"- Model:{model_name}\n"\
                            f"- Optimizer:{optimizer_name}\n"\
                            f"- Scheduler:{scheduler_name}\n"\
                            f"- LossFunc:{loss_func_name}\n"\
                            f"- Lr:{scheduler.get_last_lr()[0]:.6f}\n"\
                            f"- val_cost_time:{val_cost_time:.4f}s ⏱️\n"

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
            metrics_info = val_info_str + table_str + '\n' + loss_str  
            
            # 将指标表格写入日志文件
            custom_logger(metrics_info, logs_path)
            print(metrics_info)
            
            """------------------------------------- 保存权重文件 --------------------------------------------"""
            last_ckpt_path = os.path.join(ckpt_dir, f'{model_name}_braTS21_{loss_func_name}_{date_time_str}_{epoch}_{val_mean_loss:.4f}.pth')
            
            if val_mean_loss < best_val_loss:
                best_val_loss = val_mean_loss
                best_epoch = epoch
                with open(os.path.join(os.path.dirname(logs_path), "current_log.txt"), 'a') as f:
                    f.write(f"=== EPOCH {best_epoch} ===:\n"\
                            f"@ {date_time_str}\n"\
                            f"current lr : {scheduler.get_last_lr()[0]:.6f}\n"\
                            f"loss: Mean:{val_mean_loss:.4f}\t ET: {mean_val_et_loss:.4f}\t TC: {mean_val_tc_loss:.4f}\t WT: {mean_val_wt_loss:.4f}\n"
                            f"mean dice : {val_scores['Dice_scores'][0]:.4f}\t" \
                            f"ET : {val_scores['Dice_scores'][1]:.4f}\t"\
                            f"TC : {val_scores['Dice_scores'][2]:.4f}\t" \
                            f"WT : {val_scores['Dice_scores'][3]:.4f}\n\n")
                
                if best_val_loss < save_loss_threshold: # 损失小于0.5时保存模型
                    save_checkpoint(model, optimizer, scaler, best_epoch, best_val_loss, last_ckpt_path)
                
    print(f"😃😃😃Train finished. Best val loss: 👉{best_val_loss:.4f} at epoch {best_epoch+1}")
    # 训练完成后关闭 SummaryWriter
    writer.close() 
    



