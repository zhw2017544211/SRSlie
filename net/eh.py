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


#上采样
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


def tensor_to_numpy(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.detach().numpy()


def numpy_to_tensor(array, device=None):
    tensor = torch.from_numpy(array)
    if device:
        tensor = tensor.to(device)
    return tensor


def clahe(L, clip_limit=2.0, grid_size=(8, 8), add_one=False):

    device = L.device

    if len(L.shape) != 4 or L.shape[0] != 1 or L.shape[1] != 1:
        raise ValueError(f"输入张量应为 [1, 1, H, W]，实际为: {L.shape}")
    L_np = tensor_to_numpy(L[0, 0])
    original_min = L_np.min()
    original_max = L_np.max()
    if original_max > original_min:
        L_255 = ((L_np - original_min) / (original_max - original_min) * 255.0)
    else:
        return L

    L_255 = np.clip(L_255, 0, 255).astype(np.uint8)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced_L_255 = clahe_obj.apply(L_255)
    enhanced_L_np = ((enhanced_L_255 / 255.0) * (original_max - original_min) + original_min).astype(np.float32)
    enhanced_L = numpy_to_tensor(enhanced_L_np).unsqueeze(0).unsqueeze(0).to(device)
    if add_one:
        enhanced_L = enhanced_L + 1.0

    return enhanced_L


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class LSAModule(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv_upper = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_lower = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, semantic_feature):

        upper_feat = self.conv_upper(semantic_feature)
        lower_feat = self.conv_lower(semantic_feature)

        # 返回上方路径的结果和用于乘法的attention map
        return upper_feat, lower_feat

class SemanticEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encode(x)




