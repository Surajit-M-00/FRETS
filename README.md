FRETS: Frequency-Enhanced Residual Transformer System for Contactless SpO₂ Estimation
This repository contains the official implementation of FRETS, a lightweight frequency-aware framework for non-contact SpO₂ estimation from facial videos. FRETS combines learnable DC/AC decomposition, FFT-based spectral attention, and dual-path physiological encoding to achieve accurate and real-time SpO₂ estimation using standard RGB cameras.
Traditional remote PPG-based methods suffer from motion artifacts, illumination sensitivity, and shallow temporal modeling. FRETS addresses these challenges by jointly capturing spatial, temporal, and spectral dynamics of physiological signals extracted from facial regions.

Key Features:
Contactless SpO₂ estimation from webcam/face video
Learnable DC/AC frequency separation
FFT-based Transformer for spectral–temporal modeling
Dual-stream physiological encoder
Robust under motion & low-light conditions
Low-latency, deployment-friendly architecture

Method Overview:
The FRETS pipeline consists of:
Spatio–Temporal Map Construction
Facial landmarks are sampled and aggregated across multiple color spaces.
Learnable Frequency Separation
AC/DC components are extracted adaptively.
FFT Transformer Module
Global spectral–temporal dependencies are captured using self-attention.
Dual-Path Encoding
DC (appearance) and AC (pulsatile) signals are encoded separately.
Regression Head
A fused representation predicts continuous SpO₂ waveforms.

Performance

Evaluated on:
PURE
BH-rPPG
VIPL-HR
FRETS achieves competitive or state-of-the-art MAE/RMSE with significantly lower inference latency, making it suitable for real-time and edge deployment.

Datasets:
PURE
BH-rPPG
VIPL-HR
Due to licensing constraints, datasets are not redistributed.




