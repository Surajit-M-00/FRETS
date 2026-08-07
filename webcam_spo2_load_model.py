#!/usr/bin/env python3

import argparse
import time
from collections import deque
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

###########################################################
#  MODEL DEFINITION (FULL - FROM YOUR CODE + FIXES)
###########################################################

class FrequencyFilter(nn.Module):
    def __init__(self, in_channels):
        super(FrequencyFilter, self).__init__()
        self.low_pass = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2,
                                  groups=in_channels, bias=False)
        self.high_pass = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2,
                                   groups=in_channels, bias=False)

        with torch.no_grad():
            lp_weight = torch.ones((in_channels, 1, 5)) / 5
            hp_weight = torch.tensor([[-1, -1, 4, -1, -1]]).repeat(in_channels, 1, 1)

            self.low_pass.weight.copy_(lp_weight)
            self.high_pass.weight.copy_(hp_weight)

    def forward(self, x):
        B, C, H, W = x.shape
        x1d = x.reshape(B, C, H * W)

        lp = self.low_pass(x1d)
        hp = self.high_pass(x1d)

        return lp.reshape(B, C, H, W), hp.reshape(B, C, H, W)


class FFTTransformerBlock(nn.Module):
    def __init__(self, in_channels, embed_dim=64, num_heads=4, depth=2):
        super().__init__()
        self.pool = nn.AvgPool2d((4, 4))
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(in_channels, embed_dim)

        # FIX: batch_first=True removes the PyTorch warning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.output_proj = nn.Linear(embed_dim, in_channels)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.pool(x)
        Hp, Wp = x.shape[2], x.shape[3]

        fft = torch.fft.fft(x.float(), dim=-1).real
        seq = fft.reshape(B, C, -1).permute(0, 2, 1)

        emb = self.input_proj(seq)
        trans = self.transformer(emb)
        out = self.output_proj(trans)

        out = out.permute(0, 2, 1).reshape(B, C, Hp, Wp)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return out


class DCACConvBlock(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.depth = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1,
                               groups=input_channel)
        self.bn = nn.BatchNorm2d(input_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.depth(x)))


class SpO2Model(nn.Module):
    def __init__(self, input_channel=12, output_dim=300):
        super().__init__()

        self.freq_filter = FrequencyFilter(input_channel)
        self.fft_transform = FFTTransformerBlock(input_channel)

        self.dc_conv = DCACConvBlock(input_channel)
        self.ac_conv = DCACConvBlock(input_channel)

        self.dc_resnet = resnet18(weights=None)
        self.ac_resnet = resnet18(weights=None)

        self.dc_resnet.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.ac_resnet.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.dc_resnet = nn.Sequential(*list(self.dc_resnet.children())[:-2])
        self.ac_resnet = nn.Sequential(*list(self.ac_resnet.children())[:-2])

        self.fusion_conv = nn.Conv1d(2, 1, kernel_size=1)
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        lp, hp = self.freq_filter(x)
        enhanced = self.fft_transform(lp + hp)
        x = x + enhanced

        x_dc = self.dc_conv(x)
        x_ac = self.ac_conv(x)

        Mdc = self.dc_resnet(x_dc)
        Mac = self.ac_resnet(x_ac)

        Mdc = F.adaptive_avg_pool2d(Mdc, (1, 1)).view(x.size(0), -1)
        Mac = F.adaptive_avg_pool2d(Mac, (1, 1)).view(x.size(0), -1)

        fused = torch.stack([Mdc, Mac], dim=1)
        fused = self.fusion_conv(fused).squeeze(1)

        out = self.fc(fused)
        return out, x_dc, x_ac

###########################################################
#  LOADER (ROBUST)
###########################################################

def load_checkpoint(model, path, device):
    try:
        ck = torch.load(path, map_location=device)
    except:
        print("[ERROR] Could not load:", path)
        return False

    state = None

    if isinstance(ck, dict):
        for k in ["state_dict", "model_state", "model", "net"]:
            if k in ck:
                state = ck[k]
                break

        if state is None:
            state = ck

    # strip "module."
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v

    try:
        model.load_state_dict(new_state, strict=False)
        print("[INFO] Loaded:", path)
        return True
    except Exception as e:
        print("[ERROR] Failed to load:", e)
        return False


###########################################################
#  INPUT PIPELINE (4 frames → 12 channels)
###########################################################

def build_input(frames, H, W):
    assert len(frames) == 4
    chans = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (W, H)).astype(np.float32) / 255.0
        c = resized.transpose(2, 0, 1)
        chans.append(c)
    stacked = np.concatenate(chans, axis=0)
    return torch.from_numpy(stacked[None]).float()


###########################################################
#  SIMPLE SPO2 MAPPER (PLACEHOLDER)
###########################################################

def estimate_spo2(raw):
    val = float(raw.mean())
    return np.clip(50 + val * 50, 70, 100)


###########################################################
#  DRAW WAVEFORM
###########################################################

def draw_wave(frame, waveform, box=(10, 50, 360, 120)):
    x, y, w, h = box
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    norm = waveform - waveform.min()
    if norm.max() > 0:
        norm /= norm.max()

    ys = (h - 1 - norm * (h - 1)).astype(int)
    step = max(1, w // len(ys))

    prev = None
    for i, v in enumerate(ys):
        px = min(w - 1, i * step)
        py = v
        if prev is not None:
            cv2.line(canvas, prev, (px, py), (0, 255, 0), 1)
        prev = (px, py)

    frame[y:y + h, x:x + w] = cv2.addWeighted(frame[y:y + h, x:x + w], 0.3, canvas, 0.7, 0)
    return frame


###########################################################
#  MAIN LOOP
###########################################################

def main(args):
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    print("[INFO] Device:", device)

    model = SpO2Model().to(device).eval()

    # Try loading your files
    load_checkpoint(model, args.model_path, device)
    load_checkpoint(model, args.checkpoint, device)

    # CAMERA
    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.preview_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.preview_h)

    # Fix Qt threading
    cv2.namedWindow("SpO2 Live", cv2.WINDOW_NORMAL)
    cv2.startWindowThread()

    buf = deque(maxlen=4)
    fps_buf = deque(maxlen=30)

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        buf.append(frame.copy())
        disp = frame.copy()

        if len(buf) == 4:
            x = build_input(list(buf), 68, 300).to(device)
            out, _, _ = model(x)
            raw = out.detach().cpu().numpy().squeeze()

            spo2 = estimate_spo2(raw)
            cv2.putText(disp, f"SpO2: {spo2:.1f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            disp = draw_wave(disp, raw)

        fps = 1.0 / (time.time() - t0)
        fps_buf.append(fps)
        cv2.putText(disp, f"FPS: {np.mean(fps_buf):.1f}", (10, args.preview_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("SpO2 Live", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


###########################################################
#  CLI
###########################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str,
        default="/home/tih_isi_9/Downloads/SPO2_code_v2-main/results/Bh_rppg/Bh_rPPG_weight_1/fold_10/model_epoch_50.pth")

    parser.add_argument("--checkpoint", type=str,
        default="/home/tih_isi_9/Downloads/SPO2_code_v2-main/results/Bh_rppg/Bh_rPPG_checkpoints_1/fold_10_checkpoint.pth")

    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--preview_w", type=int, default=640)
    parser.add_argument("--preview_h", type=int, default=480)

    args = parser.parse_args()
    main(args)
