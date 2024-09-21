# -*- coding: UTF-8 -*-
'''

Describle:         计算工具：
                    1. 计算步长、卷积核尺寸和填充以满足输出尺寸要求

Created on         2024/07/31 15:40:28
Author:           @ Mr_Robot
Current State:    
'''


def calculateTransConvParam(W_in, H_in, W_out, H_out, kernel_size):
    # 假设步长是已知的，例如 stride = 2
    stride = 2
    for padding in range(kernel_size):
        if (W_out == ((W_in-1) * stride) - 2 * padding + kernel_size) and \
           (H_out == ((H_in-1) * stride) - 2 * padding + kernel_size):
            return stride, kernel_size, padding
    raise ValueError("无法找到合适的步长、卷积核尺寸和填充以满足输出尺寸要求")




if __name__ == "__main__":
    # 输入和输出尺寸
    IN_SIZE = 16
    OUT_SIZE = 32
    W_in, H_in = IN_SIZE, IN_SIZE  # 输入特征图尺寸
    W_out, H_out = OUT_SIZE, OUT_SIZE  # 输出特征图尺寸
    kernel_size = 4  # 假设卷积核尺寸是3x3

    # 计算参数
    try:
        stride, kernel_size, padding = calculateTransConvParam(W_in, H_in, W_out, H_out, kernel_size)
        print(f"步长: {stride}, 卷积核尺寸: {kernel_size}, 填充: {padding}")
    except ValueError as e:
        print(e)