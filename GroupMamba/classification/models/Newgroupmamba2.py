import torch
import torch.nn as nn
import torch.fft
from timm.models.layers import DropPath, trunc_normal_
import math
from einops import rearrange
import random
try:
    from .ss2d2 import SS2D
    from .csms6s2 import CrossScan_1, CrossScan_2, CrossScan_3, CrossScan_4
    from .csms6s2 import CrossMerge_1, CrossMerge_2, CrossMerge_3, CrossMerge_4
except:
    from ss2d2 import SS2D
    from csms6s2 import CrossScan_1, CrossScan_2, CrossScan_3, CrossScan_4
    from csms6s2 import CrossMerge_1, CrossMerge_2, CrossMerge_3, CrossMerge_4

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        mip = max(8, in_planes // ratio)
        self.avg_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, mip, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(mip, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = self.sigmoid(max_out)
        return out
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x


class GroupMambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=1, d_conv=3, expand=1, reduction=16):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.ca=ChannelAttention(input_dim)

        self.mamba_g1 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )
        self.mamba_g2 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv,
        )
        self.mamba_g3 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )
        self.mamba_g4 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )

        self.skip_scale = nn.Parameter(torch.ones(1))
        # self.conv=nn.Conv2d(input_dim, output_dim,kernel_size=1)

    def forward(self, x,y):
        B,C,H,W=x.shape
        z=self.ca(x).permute(0,2,3,1)
        x=x.permute(0,2,3,1)
        y=y.permute(0,2,3,1)

        x1, x2, x3, x4 = torch.chunk(x, 4, dim=-1)
        y1, y2, y3, y4 = torch.chunk(y, 4, dim=-1)

        x_mamba1 = self.mamba_g1(x1, y1, CrossScan=CrossScan_1, CrossMerge=CrossMerge_1)
        x_mamba2 = self.mamba_g2(x2, y2,CrossScan=CrossScan_2, CrossMerge=CrossMerge_2)
        x_mamba3 = self.mamba_g3(x3, y3, CrossScan=CrossScan_3, CrossMerge=CrossMerge_3)
        x_mamba4 = self.mamba_g4(x4, y4,CrossScan=CrossScan_4, CrossMerge=CrossMerge_4)
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=-1)
        x_mamba=(x_mamba*z).view(B,H*W,C)
        # x_mamba=x_mamba.view(B,H*W,C)
        x_mamba = self.norm(x_mamba)
        x_mamba=x_mamba.view(B,H,W,C).permute(0,3,1,2)
        return x_mamba

class Block_mamba(nn.Module):
    def __init__(self,
        dim,
        # mlp_ratio,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)

        self.attn = GroupMambaLayer(dim, dim,d_state=8)
        # self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x,y):
        x = x + self.drop_path(self.attn(x,y))
        # x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    x = torch.randn(4, 64, 96,96).to(device)  # Batch size 4, 3 channels, 416x416
    z = torch.randn(4, 64, 96,96).to(device)  # Batch size 4, 3 channels, 416x416

    model = Block_mamba(64,0.1).to(device)
    y = model(x,z)
    print("Model output shape:", y.shape)
