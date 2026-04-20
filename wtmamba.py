from torchsummary import summary

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as selective_scan_fn_v1
except:
    pass
from torch.nn import Module
from SAM2UNet import SAM2UNet
import torch.nn as nn
import torch
import torch.nn.functional as F
from GroupMamba.classification.models.Newgroupmamba2 import Block_mamba

from pytorch_wavelets import DWTForward


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class FrequencyFeatureSplitter(nn.Module):
    def __init__(self, cutoff_radius_ratio: float = 0.12):
        super().__init__()
        if not (0 < cutoff_radius_ratio < 0.5):
            raise ValueError("cutoff_radius_ratio must be between 0 and 0.5 (exclusive).")
        self.cutoff_radius_ratio = cutoff_radius_ratio

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x_fft = torch.fft.fft2(x)


        fx = torch.fft.fftfreq(W, device=x.device) * W
        fy = torch.fft.fftfreq(H, device=x.device) * H
        fx_mesh, fy_mesh = torch.meshgrid(fx, fy, indexing='xy')
        distance_from_center = torch.sqrt(fx_mesh ** 2 + fy_mesh ** 2)


        cutoff_radius = min(H, W) * self.cutoff_radius_ratio
        low_pass_mask = (distance_from_center <= cutoff_radius).float()
        low_pass_mask = low_pass_mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
        high_pass_mask = 1.0 - low_pass_mask


        filtered_low_fft = x_fft * low_pass_mask
        filtered_high_fft = x_fft * high_pass_mask


        low_freq_features = torch.fft.ifft2(filtered_low_fft).real
        high_freq_features = torch.fft.ifft2(filtered_high_fft).real
        return low_freq_features, high_freq_features


class LowFreqFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.se_block = ChannelAttention(in_channels * 2)

        self.conv_fuse = nn.Conv2d(in_channels * 2, in_channels, 1)

    def forward(self, freq_low, wave_LL):

        wave_LL_upsampled = F.interpolate(wave_LL, size=freq_low.shape[2:], mode='bilinear', align_corners=False)


        combined_features = torch.cat([wave_LL_upsampled, freq_low], dim=1)


        attention_weights = self.se_block(combined_features)
        fused_features = combined_features * attention_weights


        fused_features = self.conv_fuse(fused_features)
        return fused_features


class HighFreqFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_combine_wavelets = nn.Conv2d(in_channels * 3, in_channels, 1,groups=in_channels)



        self.spatial_attention = SpatialAttention()

        self.conv_fuse = nn.Conv2d(in_channels * 2, in_channels, 1)

    def forward(self, freq_high, LH, HL, HH):
        LH_high_upsampled = F.interpolate(LH, size=freq_high.shape[2:], mode='bilinear', align_corners=False)
        HL_high_upsampled = F.interpolate(HL, size=freq_high.shape[2:], mode='bilinear', align_corners=False)
        HH_high_upsampled = F.interpolate(HH, size=freq_high.shape[2:], mode='bilinear', align_corners=False)

        combined_wavelets = torch.cat([LH_high_upsampled, HL_high_upsampled, HH_high_upsampled], dim=1)
        combined_wavelets = self.conv_combine_wavelets(combined_wavelets)


        combined_features = torch.cat([freq_high, combined_wavelets], dim=1)


        attention_weights = self.spatial_attention(combined_features)
        fused_features = combined_features * attention_weights


        fused_features = self.conv_fuse(fused_features)
        return fused_features


class FrequencyEnhancement(nn.Module):
    def __init__(self, in_channels: int, cutoff_radius_ratio: float = 0.08):
        super().__init__()
        self.freq_splitter = FrequencyFeatureSplitter(cutoff_radius_ratio=cutoff_radius_ratio)
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')

        self.low_freq_fusion = LowFreqFusion(in_channels)
        self.high_freq_fusion = HighFreqFusion(in_channels)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        freq_low, freq_high = self.freq_splitter(x)  # 形状: (B, C, H, W)

        wave_LL = self.dwt(x)[0]
        wave_high_bands_stacked = self.dwt(x)[1][0]

        LH, HL, HH = torch.split(wave_high_bands_stacked, 1, dim=2)
        LH = LH.squeeze(2)
        HL = HL.squeeze(2)
        HH = HH.squeeze(2)
        low_freq_fused = self.low_freq_fusion(freq_low, wave_LL)
        high_freq_fused = self.high_freq_fusion(freq_high, LH, HL, HH)
        fre = torch.cat([low_freq_fused, high_freq_fused], dim=1)

        return fre


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False,groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class MAMBA(nn.Module):
    def __init__(self, dim, drop_path=0.1):
        super().__init__()
        self.conv = nn.Conv2d(3 * dim, dim, kernel_size=3, padding=1,groups=dim)

        self.bn = nn.BatchNorm2d(dim)
        self.mamba1 = Block_mamba(dim, drop_path=drop_path)
        self.down_conv = nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False,groups=dim//2)

    def forward(self, x1, x2):
        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.conv(x1)
        x1 = self.bn(x1)
        x1 = self.mamba1(x2,x1)
        x = self.down_conv(x1)
        return x


class decoder(nn.Module):
    def __init__(self, dims_decoder, num_output_channels=1):
        super().__init__()
        C4, C3, C2, C1 = dims_decoder
        self.mamba1 = MAMBA(C1)
        self.mamba2 = MAMBA(C2)
        self.mamba3 = MAMBA(C3)
        self.mamba4 = MAMBA(C4)
        self.upsample_conv3 = UpsampleBlock(C3, C3)
        self.upsample_conv2 = UpsampleBlock(C2, C2)
        self.upsample_conv1 = UpsampleBlock(C1, C1)
    def forward(self, features, original_x4):
        f1, f2, f3, f4 = features
        x4 = self.mamba4(f4, original_x4)
        x4 = self.upsample_conv3(x4)

        x3 = self.mamba3(f3, x4)
        x3 = self.upsample_conv2(x3)

        x2 = self.mamba2(f2, x3)
        x2 = self.upsample_conv1(x2)

        x1 = self.mamba1(f1, x2)
        return x1


class Decoder(Module):
    def __init__(self, dims_decoder=[768, 384, 192, 96]):
        super(Decoder, self).__init__()
        self.fre1 = FrequencyEnhancement(dims_decoder[3])
        self.fre2 = FrequencyEnhancement(dims_decoder[2])  # C2
        self.fre3 = FrequencyEnhancement(dims_decoder[1])  # C3
        self.fre4 = FrequencyEnhancement(dims_decoder[0])
        self.decoder = decoder(dims_decoder, num_output_channels=dims_decoder[3] // 4)  # C1 // 4

    def forward(self, features):
        # 完整网络
        f1_orig, f2_orig, f3_orig, f4_orig = features  # 为了获取 H, W 和传入原始 x4，这里使用 _orig 后缀
        x4_original_input = f4_orig

        f1 = self.fre1(f1_orig)
        f2 = self.fre2(f2_orig)
        f3 = self.fre3(f3_orig)
        f4 = self.fre4(f4_orig)
        enhanced_features = (f1, f2, f3, f4)
        decoded_output = self.decoder(enhanced_features, x4_original_input)  # 传入原始x4 (f4_orig)
        return decoded_output

class net(nn.Module):
    def __init__(self,**kwargs):
        super(net, self).__init__()
        checkpoint_path='sam2.pt'
        self.backbone = SAM2UNet(checkpoint_path=checkpoint_path)
        self.decode_head = Decoder(dims_decoder=[1152, 576, 288, 144])#l
        self.conv=nn.Conv2d(72,1,kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        f1 = self.decode_head(features)
        x4=self.conv(f1)
        x4 = F.interpolate(x4, size=x.shape[2:], mode='bilinear', align_corners=False)
        return x4


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    x = torch.randn(1, 3, 384, 384).to(device)
    model = net().to(device)
    y = model(x)
    print("Model output shape:", y.shape)