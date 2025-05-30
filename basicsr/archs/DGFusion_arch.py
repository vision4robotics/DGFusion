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
        out += self.identity(x)

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

        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.ones(1))
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

        self.amp_sub1 = nn.Sequential(nn.Conv2d(4 * channels, 4 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(4 * channels, 4 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False))
        self.pha_sub1 = nn.Sequential(nn.Conv2d(4 * channels, 4 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(4 * channels, 4 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False))
        self.amp_sub2 = nn.Sequential(nn.Conv2d(4 * channels, 4 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(4 * channels, 4 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False))
        self.pha_sub2 = nn.Sequential(nn.Conv2d(4 * channels, 4 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(4 * channels, 4 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False))

        self.conv1s = nn.Sequential(nn.Conv2d(4 * channels, 4 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
                                    nn.Conv2d(4 * channels, 4 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False))
        self.conv2s = nn.Sequential(nn.Conv2d(4 * channels, 4 * channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=False),
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
        amp_fuse = self.conv1(self.amp_fuse(torch.cat([visF_amp, irF_amp], 1)) + torch.cat([visF_amp, irF_amp], 1))
        pha_fuse = self.conv2(self.pha_fuse(torch.cat([visF_pha, irF_pha], 1)) + torch.cat([visF_pha, irF_pha], 1))
        subF = self.a * visF - self.b * irF
        amp_sub = torch.abs(subF)
        pha_sub = torch.angle(subF)
        amp_sub = self.amp_sub2(self.amp_sub1(amp_sub)) + amp_sub
        pha_sub = self.pha_sub2(self.pha_sub1(pha_sub)) + pha_sub
        real_sub = amp_sub * torch.cos(pha_sub)
        imag_sub = amp_sub * torch.sin(pha_sub)
        subF = torch.cat([real_sub, imag_sub], 1)
        real_aps = amp_fuse * torch.cos(pha_sub)
        imag_aps = amp_fuse * torch.sin(pha_sub)
        real_asp = amp_sub * torch.cos(pha_fuse)
        imag_asp = amp_sub * torch.sin(pha_fuse)
        apsF = torch.cat([real_aps, imag_aps], 1)  #####
        aspF = torch.cat([real_asp, imag_asp], 1)  #####

        return apsF, aspF, subF


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
            nn.Conv2d(8 * channels, 8 * channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8 * channels, 8 * channels, 3, 1, 1)
        )
        self.norm3 = LayerNorm(4 * channels, LayerNorm_type='WithBias')
        self.norm4 = LayerNorm(4 * channels, LayerNorm_type='WithBias')

    def forward(self, subF, apsF, aspF):
        F1 = self.wq(subF) * self.wk(aspF)
        C_F1 = F1.size(1) // 2
        F1_real = F1[:, :C_F1, :, :]
        F1_imag = F1[:, C_F1:, :, :]
        F1 = torch.complex(F1_real, F1_imag)
        F1 = torch.fft.ifft2(F1).real
        F1 = self.norm1(F1)

        F2 = self.wv(apsF)
        C_F2 = F2.size(1) // 2
        F2_real = F2[:, :C_F2, :, :]
        F2_imag = F2[:, C_F2:, :, :]
        F2 = torch.complex(F2_real, F2_imag)
        F2 = torch.fft.ifft2(F2).real
        F2 = self.norm2(F2)
        C_subF = subF.size(1) // 2
        subF_real = subF[:, :C_subF, :, :]
        subF_imag = subF[:, C_subF:, :, :]
        subF = torch.complex(subF_real, subF_imag)
        subf = torch.fft.ifft2(subF).real
        result = F1 * F2 + subf
        result = self.norm3(result)
        return result

class interaction(nn.Module):
    def __init__(self, channels):
        super(interaction, self).__init__()

        self.subF_att = AttentionModule(channels)
        self.fuse_att = AttentionModule(channels)
        self.fuse = nn.Sequential(nn.Conv2d(8*channels, 4*channels, 3, 1, 1), nn.Conv2d(4*channels, 8*channels, 3, 1, 1), nn.Sigmoid())


    def forward(self,apsF, aspF, subF):

        fre = self.subF_att(subF, apsF, aspF)
        spa = self.fuse_att(subF, aspF, apsF)
        fuse = self.fuse(torch.cat((fre, spa), 1))
        fre_a, spa_a = fuse.chunk(2, dim=1)
        spa = spa_a * spa
        fre = fre * fre_a
        res = fre + spa

        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res



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

@ARCH_REGISTRY.register()
class DGFusion(nn.Module):
    def __init__(self, vischannels, irchannels, channels):
        super(DGFusion, self).__init__()
        self.freprocess = Freprocess(vischannels, irchannels, channels)
        self.attention_module = AttentionModule(channels)
        self.interaction=interaction(channels)
        self.final_conv = nn.Sequential(nn.Conv2d(4*channels, 2*channels, 3, 1, 1),nn.LeakyReLU(0.1, inplace=False),
                                        nn.Conv2d(2*channels, 1, 3, 1, 1),nn.LeakyReLU(0.1, inplace=False))

    def forward(self, image_vis, image_ir):
        visf = image_vis[:, :1]
        irf = image_ir
        apsF, aspF, subF = self.freprocess(visf, irf)
        result = self.interaction(apsF, aspF, subF)
        fused = self.final_conv(result)

        return fused
