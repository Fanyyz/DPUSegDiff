# Diffusion Model for Medical Image Segmentation

This repository contains an implementation of a **Denoising Probabilistic Diffusion Model (DPUSegDiff)** for medical image segmentation. The model integrates **CNN, Transformer, edge detection, feature fusion, and diffusion model denoising** techniques within a **dual-path network** to improve segmentation performance.

## Features
- **Diffusion-based segmentation**: Uses a probabilistic diffusion model for robust segmentation.
- **Dual-path architecture**: Combines a **CNN path** (for local feature extraction) and a **Transformer path** (for global feature modeling).
- **Feature fusion techniques**:
  - Sobel-based **local edge information fusion** (CNN path).
  - **Cross-self-attention fusion** (Transformer path).
- **Multi-dataset support**: Evaluated on **ISIC 2018, PH2, BraTS, ClinicDB, and CVC-ClinicDB** datasets.

---
pip install -r requirements.txt
## Train
python train.py --dataset ISIC2018 --batch_size 16 --epochs 100 --lr 0.0001

@article{your_paper,
  title={Diffusion-Based Medical Image Segmentation},
  author={Your Name et al.},
  journal={ArXiv},
  year={2025}
}
