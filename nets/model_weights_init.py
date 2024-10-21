# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/09/28 10:57:51
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 模型权重初始化
=================================================
'''

from torch import nn


"""---------------------------------------- 权重初始化 ----------------------------------------------"""
def init_weights_pro(model, init_type='normal', activation='relu', init_gain=0.02, always_init=False):
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            if init_type == 'kaiming_normal':
                if isinstance(m, nn.ConvTranspose3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=activation)  # `fan_in` for ConvTranspose3d
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)  # `fan_out` for Conv3d
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.normal_(m.weight, 0, init_gain)
            
            if always_init or not (init_type in ['kaiming_normal', 'xavier_normal']):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm3d):
            if always_init or init_type in ['kaiming_normal', 'xavier_normal']:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            if init_type == 'kaiming_normal':
                if isinstance(m, nn.ConvTranspose3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=activation)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.normal_(m.weight, 0, init_gain)
                
            if always_init:
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def init_weights_light(model,  init_gain=0.02):
    for m in model.modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            m.weight =  nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=init_gain)
            if m.bias is not None:
                m.bias =  nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.BatchNorm3d):
        #     nn.init.constant_(m.weight, 1)
        #     nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.Linear):
        #     nn.init.normal_(m.weight, 0, init_gain)
        #     nn.init.constant_(m.bias, 0)



# From https://github.com/MrGiovanni/UNetPlusPlus/blob/master/pytorch/nnunet/network_architecture/initialization.py
class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class InitWeights_XavierUniform(object):
    def __init__(self, gain=1):
        self.gain = gain

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.xavier_uniform_(module.weight, self.gain)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)