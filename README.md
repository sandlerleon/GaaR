# Privacy-Preserving Local AI: Evolutionary NAS on CIFAR-10 using GaaR

Bio-inspired Neural Architecture Search achieving **94.36% CIFAR-10 accuracy** 
in **45 minutes** across **696 generations** on a single AMD Radeon RX 5700 GPU.

## Overview
Evolutionary NAS framework using three bio-inspired control mechanisms:
- **GaaR Network** — Goal-Aware Adaptive Regulation
- **Banana Core** — homeostatic parameter stabilisation  
- **StruggleLayer** — adaptive hardening under noisy evaluation

Runs entirely on local hardware with zero data egress. Supports AMD 
DirectML, Apple Silicon MPS, NVIDIA CUDA, and CPU.

## Files
- `cifar_evolution_engine_local.py` — main evolutionary search loop
- `train_production_champion.py` — 200-epoch production training
- `evaluate_production_champion.py` — confusion matrix evaluation

## Paper & Data
- Zenodo (code + model + paper): [10.5281/zenodo.20737711](https://zenodo.org/records/20737711)

## arXiv Submission
This work is pending arXiv submission in cs.NE and cs.CV.
If you are an established arXiv author in either category and find 
this work interesting, I would be grateful for an endorsement:

https://arxiv.org/auth/endorse?x=H9WTH6

— Leon Sandler
