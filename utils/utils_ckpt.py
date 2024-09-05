# -*- coding: UTF-8 -*-
'''

Describle:         加载预训练模型权重、优化器状态、损失函数状态等

Created on         2024/07/31 15:29:37
Author:           @ Mr_Robot
Current State:    
'''

import torch

def load_checkpoint(model, optimizer, scaler, checkpoint_path):
    best_val_loss = float('inf')
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(f"***Resuming training from epoch {start_epoch}...")
    return model, optimizer, scaler, start_epoch, best_val_loss

def save_checkpoint(model, optimizer, scaler, epoch, best_val_loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict()}
    
    torch.save(checkpoint, checkpoint_path)
    print(f"***Saving checkpoint to {checkpoint_path}...")

    
    