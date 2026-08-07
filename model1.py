import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np
from torchvision.models import resnet18
from torchvision.utils import save_image


# =========================
# ST-MAP SAVING UTILITY
# =========================
def save_st_map(tensor, save_dir, name, batch_idx=0, channel_idx=0):
    """
    tensor: [B, C, H, W]
    Saves one ST-map image
    """
    os.makedirs(save_dir, exist_ok=True)

    st_map = tensor[batch_idx, channel_idx].detach().cpu()

    # Normalize to [0, 1]
    st_map = (st_map - st_map.min()) / (st_map.max() - st_map.min() + 1e-8)

    save_image(st_map.unsqueeze(0), os.path.join(save_dir, f"{name}.png"))


# =========================
# Frequency Filter Module
# =========================
class FrequencyFilter(nn.Module):
    def __init__(self, in_channels):
        super(FrequencyFilter, self).__init__()

        self.low_pass = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=5, padding=2,
            groups=in_channels, bias=False
        )
        self.high_pass = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=5, padding=2,
            groups=in_channels, bias=False
        )

        with torch.no_grad():
            lp_weight = torch.ones((in_channels, 1, 5)) / 5
            hp_weight = torch.tensor([[-1, -1, 4, -1, -1]]).repeat(in_channels, 1, 1)
            self.low_pass.weight.copy_(lp_weight)
            self.high_pass.weight.copy_(hp_weight)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)

        lp = self.low_pass(x)
        hp = self.high_pass(x)

        return lp.reshape(B, C, H, W), hp.reshape(B, C, H, W)


# =========================
# FFT + Transformer Block
# =========================
class FFTTransformerBlock(nn.Module):
    def __init__(self, in_channels, embed_dim=64, num_heads=4, depth=2):
        super(FFTTransformerBlock, self).__init__()

        self.pool = nn.AvgPool2d(kernel_size=(4, 4))
        self.input_proj = nn.Linear(in_channels, embed_dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=False
            ),
            num_layers=depth
        )

        self.output_proj = nn.Linear(embed_dim, in_channels)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.pool(x)
        H_, W_ = x.shape[2], x.shape[3]

        x_fft = torch.fft.fft(x.float(), dim=-1).real
        x_fft = x_fft.reshape(B, C, -1).permute(0, 2, 1)

        x_proj = self.input_proj(x_fft)
        x_proj = x_proj.permute(1, 0, 2)

        x_trans = self.transformer(x_proj)

        x_out = self.output_proj(x_trans.permute(1, 0, 2))
        x_out = x_out.permute(0, 2, 1).reshape(B, C, H_, W_)

        x_out = F.interpolate(x_out, size=(H, W), mode='bilinear', align_corners=False)

        return x_out


# =========================
# DC / AC Conv Block
# =========================
class DCACConvBlock(nn.Module):
    def __init__(self, input_channel):
        super(DCACConvBlock, self).__init__()

        self.depthwise_conv = nn.Conv2d(
            input_channel, input_channel,
            kernel_size=3, padding=1,
            groups=input_channel
        )
        self.bn = nn.BatchNorm2d(input_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# =========================
# SpO2 MODEL (AUTO SAVE)
# =========================
class SpO2Model(nn.Module):
    def __init__(self, input_channel=12, output_dim=300,
                 save_st_maps=False, save_dir="st_maps"):
        super(SpO2Model, self).__init__()

        self.save_st_maps = save_st_maps
        self.save_dir = save_dir
        self._saved_once = False

        self.freq_filter = FrequencyFilter(input_channel)
        self.fft_transform = FFTTransformerBlock(input_channel)

        self.dc_conv = DCACConvBlock(input_channel)
        self.ac_conv = DCACConvBlock(input_channel)

        self.dc_resnet = resnet18(pretrained=False)
        self.ac_resnet = resnet18(pretrained=False)

        self.dc_resnet.conv1 = nn.Conv2d(
            input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.ac_resnet.conv1 = nn.Conv2d(
            input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.dc_resnet = nn.Sequential(*list(self.dc_resnet.children())[:-2])
        self.ac_resnet = nn.Sequential(*list(self.ac_resnet.children())[:-2])

        self.fusion_conv = nn.Conv1d(2, 1, kernel_size=1)
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        # ---------- RAW ST-MAP ----------
        if self.save_st_maps and not self._saved_once:
            save_st_map(x, self.save_dir, "raw_st_map")

        # ---------- Frequency Filter ----------
        x_lp, x_hp = self.freq_filter(x)
        x_filtered = x_lp + x_hp

        if self.save_st_maps and not self._saved_once:
            save_st_map(x_lp, self.save_dir, "low_pass_st_map")
            save_st_map(x_hp, self.save_dir, "high_pass_st_map")
            save_st_map(x_filtered, self.save_dir, "filtered_st_map")

        # ---------- FFT + Transformer ----------
        x_enhanced = self.fft_transform(x_filtered)

        if self.save_st_maps and not self._saved_once:
            save_st_map(x_enhanced, self.save_dir, "fft_enhanced_st_map")
            self._saved_once = True

        # Residual
        x = x + x_enhanced

        # ---------- DC / AC ----------
        x_dc = self.dc_conv(x)
        x_ac = self.ac_conv(x)

        M_dc = self.dc_resnet(x_dc)
        M_ac = self.ac_resnet(x_ac)

        M_dc = F.adaptive_avg_pool2d(M_dc, (1, 1)).view(x.size(0), -1)
        M_ac = F.adaptive_avg_pool2d(M_ac, (1, 1)).view(x.size(0), -1)

        fused = self.fusion_conv(torch.stack([M_dc, M_ac], dim=1)).squeeze(1)
        output = self.fc(fused)

        return output, x_dc, x_ac


# =========================
# TESTING
# =========================
if __name__ == "__main__":
    model = SpO2Model(
        input_channel=12,
        output_dim=300,
        save_st_maps=True,
        save_dir="st_maps_example"
    )

    dummy_input = torch.randn(8, 12, 68, 300)
    output, x_dc, x_ac = model(dummy_input)

    print("Output shape:", output.shape)
    print("DC ST-map shape:", x_dc.shape)
    print("AC ST-map shape:", x_ac.shape)

