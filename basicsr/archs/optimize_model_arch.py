# -*- coding: utf-8 -*-
from basicsr.utils.registry import ARCH_REGISTRY
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
# from models.utils.CDC import cdcconv
import cv2
import os
from einops import rearrange
import numbers


# import common


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)  # 残差连接

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d=1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class Freprocess(nn.Module):
    def __init__(self, vischannels, irchannels, channels):
        super(Freprocess, self).__init__()
        self.a = nn.Parameter(torch.ones(1))  # 初始化为 1
        self.b = nn.Parameter(torch.ones(1))  # 初始化为 1
        self.pre1 = nn.Sequential(nn.Conv2d(vischannels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                                  nn.Conv2d(channels, 2 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                                  nn.Conv2d(2 * channels, 4 * channels, 3, 1, 1))
        self.pre2 = nn.Sequential(nn.Conv2d(irchannels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                                  nn.Conv2d(channels, 2 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                                  nn.Conv2d(2 * channels, 4 * channels, 3, 1, 1))

        self.amp_fuse = nn.Sequential(nn.Conv2d(8 * channels, 8 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(8 * channels, 8 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False))
        self.pha_fuse = nn.Sequential(nn.Conv2d(8 * channels, 8 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(8 * channels, 8 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False))
        self.amp_res = nn.Sequential(nn.Conv2d(8 * channels, 4 * channels, 3, 1, 1))
        self.pha_res = nn.Sequential(nn.Conv2d(8 * channels, 4 * channels, 3, 1, 1))

        self.conv1 = nn.Sequential(nn.Conv2d(8 * channels, 4 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                                   nn.Conv2d(4 * channels, 4 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False))
        self.conv2 = nn.Sequential(nn.Conv2d(8 * channels, 4 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                                   nn.Conv2d(4 * channels, 4 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False))

        self.post = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, visf, irf):
        _, _, H, W = visf.shape
        visF = torch.fft.fft2(self.pre1(visf), norm='backward')
        irF = torch.fft.fft2(self.pre2(irf), norm='backward')

        visF_amp = torch.abs(visF)
        visF_pha = torch.angle(visF)
        irF_amp = torch.abs(irF)
        irF_pha = torch.angle(irF)

        amp_Fuse = self.conv1(self.amp_fuse(torch.cat([visF_amp, irF_amp], 1)) + torch.cat([visF_amp, irF_amp], 1))
        pha_Fuse = self.conv2(self.pha_fuse(torch.cat([visF_pha, irF_pha], 1)) + torch.cat([visF_pha, irF_pha], 1))

        real_Fuse = amp_Fuse * torch.cos(pha_Fuse)
        imag_Fuse = amp_Fuse * torch.sin(pha_Fuse)
        Fuse=torch.complex(real_Fuse, imag_Fuse)

        subF = self.a * visF - self.b*irF

        return Fuse, subF



class AttentionModule(nn.Module):
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        self.norm1 = LayerNorm(4 * channels, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(4 * channels, LayerNorm_type='WithBias')
        # self.norm1 = LayerNorm(4*channels, LayerNorm_type='WithBias')
        # self.norm2 = LayerNorm(4*channels, LayerNorm_type='WithBias')
        self.wq = nn.Sequential(
            nn.Conv2d(8 * channels, 8 * channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8 * channels, 8 * channels, 3, 1, 1)
        )
        self.wk = nn.Sequential(
            nn.Conv2d(8 * channels, 8 * channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8 * channels, 8 * channels, 3, 1, 1)
        )
        self.wv = nn.Sequential(
            nn.Conv2d(4 * channels, 4 * channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(4 * channels, 4 * channels, 3, 1, 1)
        )

    def forward(self, Fuse,subF):

        attenF = subF * Fuse
        attenf = torch.fft.ifft2(attenF).real  # 逆傅里叶变换
        attenf = self.norm1(attenf)  # 归一化

        fuse=torch.fft.ifft2(Fuse).real
        fuse1 = self.norm2(self.wv(fuse))


        result = attenf * fuse1 + torch.fft.ifft2(subF)
        result = result.real  # F1和F2相乘后与sub相加
        # print("result",result.shape)

        return result


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()


# 其他初始化函数和模块（如 UNetConvBlock, DenseBlock, Freprocess, AttentionModule）保持不变...

@ARCH_REGISTRY.register()
class optimize_model(nn.Module):
    def __init__(self, vischannels, irchannels, channels):
        super(optimize_model, self).__init__()
        self.freprocess = Freprocess(vischannels, irchannels, channels)
        # self.spatialprocess=Spatialprocess(vischannels, irchannels, channels)
        # self.interacte = Interacte(vischannels, irchannels, channels)
        self.attention_module = AttentionModule(channels)
        # self.final_conv = nn.Conv2d(channels, 3, kernel_size=1)  # 将输出转换为3通道图像
        self.final_conv = nn.Sequential(nn.Conv2d(4 * channels, 2 * channels, 3, 1, 1),
                                        nn.LeakyReLU(0.1, inplace=False),
                                        nn.Conv2d(2 * channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                                        nn.Conv2d(channels, 1, 3, 1, 1),
                                        nn.LeakyReLU(0.1, inplace=False))  # 将输出转换为3通道图像

    def forward(self, image_vis, image_ir):
        visf = image_vis[:, :1]
        irf = image_ir
        Fuse, subF = self.freprocess(visf, irf)
        result = self.attention_module(Fuse,subF)
        fused = self.final_conv(result)

        return fused
