import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.vssd_torch import Backbone_VMAMBA2
import torch.fft
import pywt
import math
import cv2
from einops import rearrange
from net.torchnssd import (
    Backbone_VMAMBA2
)


class upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsampling, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                       output_padding=1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out


class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class Self_Attention(nn.Module):  #  1 c h w
    def __init__(self, dim, num_heads, bias):
        super(Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=(3, 3), stride=(1, 1),
                                    padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(Cross_Attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.value = Depth_conv(in_ch=dim, out_ch=dim)
        self.dropout = nn.Dropout(dropout)
    def transpose_for_scores(self, x):
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ctx):  #  hidden查询ctx   增强ctx
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(ctx)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()

        return ctx_layer



class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_block, self).__init__()
        sequence = []
        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        ]
        self.model = nn.Sequential(*sequence)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)
    def forward(self, x):
        out = self.model(x) + self.conv(x)
        return out


class Retinex_decom(nn.Module):
    def __init__(self, channels):
        super(Retinex_decom, self).__init__()
        self.conv0 = nn.Conv2d(512, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.blocks0 = nn.Sequential(Res_block(channels, channels),
                                     Res_block(channels, channels))
        self.conv1 = nn.Conv2d(512, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.blocks1 = nn.Sequential(Res_block(channels, channels),
                                     Res_block(channels, channels))
        self.cross_attention = Cross_Attention(dim=channels, num_heads=8)
        self.self_attention = Self_Attention(dim=channels, num_heads=8, bias=True)
        self.conv0_1 = nn.Sequential(Res_block(channels, channels),
                                     nn.Conv2d(channels, 512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.conv1_1 = nn.Sequential(Res_block(channels, channels),
                                     nn.Conv2d(channels, 512, kernel_size=(3, 3), stride=(1, 1), padding=1))

    def forward(self, init_illumination, init_reflectance):
        #  l 1 512 h w -1 512 h w
        #  r 1 512 h w -1 512 h w

        Reflectance, Illumination = (self.blocks0(self.conv0(init_reflectance)),
                                     self.blocks1(self.conv1(init_illumination)))

        Reflectance_final = self.cross_attention(Illumination, Reflectance)

        Illumination_content = self.self_attention(Illumination)

        Reflectance_final = self.conv0_1(Reflectance_final + Illumination_content)
        Illumination_final = self.conv1_1(Illumination - Illumination_content)

        R = torch.sigmoid(Reflectance_final)
        L = torch.sigmoid(Illumination_final)
        # L = torch.cat([L for i in range(3)], dim=1)

        return L, R


def isigmoid(y):
    y = torch.clamp(y, min=1e-6, max=1 - 1e-6)
    return torch.log(y / (1 - y))


def normalize_brightness(image, target_brightness=64):

    device = image.device
    image = image.float()
    r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    current_brightness = 0.2989 * r + 0.5870 * g + 0.1140 * b
    current_brightness = current_brightness.mean(dim=[1, 2], keepdim=True)
    adjustment_factor = target_brightness / (current_brightness + 1e-5)

    r = torch.clamp(r * adjustment_factor, 0, 255)
    g = torch.clamp(g * adjustment_factor, 0, 255)
    b = torch.clamp(b * adjustment_factor, 0, 255)

    adjusted_image = torch.stack([r, g, b], dim=1)  # (B, C, H, W)

    return adjusted_image.to(device).float()


# class PatchExpanding(nn.Module):
#     def __init__(self, dim, scale=2):
#         super().__init__()
#         self.input_dim = dim
#         self.output_dim = dim // 2
#         self.scale = scale

#         self.linear = nn.Conv2d(dim, self.output_dim * (scale ** 2), kernel_size=1)
#
#     def forward(self, x):
#         """
#         输入: x - (B, C, H, W)
#         输出: (B, C//2, H*scale, W*scale)
#         """
#         x = self.linear(x)  # (B, output_dim * scale^2, H, W)
#
#         # (B, output_dim, H*scale, W*scale)
#         x = F.pixel_shuffle(x, upscale_factor=self.scale)
#
#         return x
class PatchExpanding(nn.Module):
    def __init__(self, dim, scale=2):
        super().__init__()
        self.output_dim = dim // 2
        self.linear = nn.Conv2d(dim, self.output_dim * scale**2, kernel_size=1)
        self.residual = nn.Sequential(
            nn.Conv2d(dim, self.output_dim, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x_ps = self.linear(x)
        x_ps = F.pixel_shuffle(x_ps, upscale_factor=2)
        x_res = self.residual(x)
        return x_ps + x_res

class StemDecoder(nn.Module):
    def __init__(self, in_chans=3, embed_dim=64):
        super().__init__()
        self.inv_conv3 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim*4, kernel_size=1, bias=False),
            nn.ConvTranspose2d(embed_dim*4, embed_dim//2,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.inv_res2 = nn.Sequential(
            nn.Conv2d(embed_dim//2, embed_dim//2, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim//2, embed_dim//2, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.inv_conv1 = nn.ConvTranspose2d(
            embed_dim//2, in_chans,
            kernel_size=3, stride=2,
            padding=1, output_padding=1, bias=False
        )

    def forward(self, x):
        # x: (B, C=embed_dim, H, W)
        x = self.inv_conv3(x)      # (B, E//2, H*2, W*2)
        x = self.inv_res2(x)       # (B, E//2, H*2, W*2)
        x = self.inv_conv1(x)      # (B, in_chans, H*4, W*4)
        return x


