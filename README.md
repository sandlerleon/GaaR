# Privacy-Preserving Local AI: Evolutionary NAS on CIFAR-10 using GaaR

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20737711.svg)](https://doi.org/10.5281/zenodo.20737711)
[![arXiv](https://img.shields.io/badge/arXiv-cs.NE%20%7C%20cs.CV-pending-red)](https://arxiv.org/auth/endorse?x=H9WTH6)

Bio-inspired Neural Architecture Search achieving **94.36% CIFAR-10 accuracy**
in **45 minutes** across **696 generations** on a single consumer AMD Radeon RX 5700 GPU —
no cloud, no data egress, no expensive hardware required.

---

## Overview

This framework performs evolutionary Neural Architecture Search entirely on local
consumer hardware using three bio-inspired control mechanisms:

| Mechanism | Role |
|---|---|
| **GaaR Network** | Goal-Aware Adaptive Regulation — controls exploit vs explore regime |
| **Banana Core** | Homeostatic parameter stabilisation — prevents hyperparameter drift |
| **StruggleLayer** | Adaptive hardening under noisy evaluation — increases resilience over generations |

The evolutionary loop runs indefinitely, logging every generation to a local SQLite
database. Local AI assistants (Ollama, LM Studio, Claude) can query this database
to suggest mutations and analyse experiment history — all without sending data to the cloud.

---

## Key Results

| Metric | Value |
|---|---|
| Search generations | 696 |
| Search time | 45 minutes |
| Hardware | AMD Radeon RX 5700 (8GB VRAM, consumer GPU) |
| CIFAR-10 test accuracy | **94.36%** |
| Training after search | 200 epochs on discovered architecture |
| Data egress | Zero |

---

## Hardware Support

| Backend | Hardware |
|---|---|
| DirectML | AMD GPUs on Windows (RX 5000 series and newer) |
| MPS | Apple Silicon (M1 / M2 / M3 / M4 / M5) |
| CUDA | NVIDIA GPUs |
| CPU | Any machine — slower but fully functional |

---

## Requirements

- Python 3.9 or newer
- Windows 10 / 11 (for DirectML), macOS (for MPS), or Linux (for CUDA/CPU)
- ~2 GB disk space for CIFAR-10 dataset (auto-downloaded on first run)
- AMD GPU users: 4GB+ VRAM recommended

Install dependencies:

```powershell
pip install torch torchvision
```

AMD GPU on Windows — also install:

```powershell
pip install torch-directml
```

NVIDIA GPU — install the CUDA version of PyTorch from:
https://pytorch.org/get-started/locally/

---

## Usage — 3 Steps in Sequence

Run these three commands in order from your project folder in PowerShell.
Each step builds on the output of the previous one.

---

### Step 1 — Run the Evolutionary Search

```powershell
python cifar_evolution_engine_local.py
```

**What it does:**
- Starts the evolutionary NAS loop — runs indefinitely until you stop it
- Each generation trains a candidate architecture on CIFAR-10 and scores it
- Results are logged to `pow_full.db` (SQLite) after every generation
- The GaaR Network, Banana Core, and StruggleLayer adapt hyperparameters automatically
- Watch the `Accuracy` and `Progress` columns in the console output

**When to stop it:**
- Let it run for at least 200-300 generations for meaningful results
- Stop it with `Ctrl + C` when accuracy stops improving across many generations
- The best configuration found is saved to the database automatically

**Console output looks like:**
```
[09:15:00] [exploit_best] Local Step=22 LR=0.0019 Batch=64 Accuracy=46.09% | Progress=30.10%
[09:15:10] [explore_mutation] Local Step=23 LR=0.0016 Batch=60 Accuracy=44.26% | Progress=30.37%
```

---

### Step 2 — Train the Champion Architecture

```powershell
python train_production_champion.py
```

**What it does:**
- Reads the best architecture configuration found in Step 1 from the database
- Trains it fully for 200 epochs with proper learning rate scheduling
- Saves the trained model weights to `champion_converged_resnet.pth`
- This is the production-quality training run — much longer than the search evaluations

**Expected duration:**
- 30-90 minutes depending on your GPU
- Console will show epoch-by-epoch accuracy climbing toward the final result

**Output file:**
```
champion_converged_resnet.pth   (saved model weights, ~9 MB)
```

---

### Step 3 — Evaluate the Champion

```powershell
python evaluate_production_champion.py
```

**What it does:**
- Loads `champion_converged_resnet.pth` from Step 2
- Runs inference on the full CIFAR-10 test set (10,000 images)
- Prints final test accuracy and a per-class confusion matrix
- Shows which of the 10 CIFAR-10 classes the model handles best and worst

**Expected output:**
```
Loading champion model...
Running evaluation on 10,000 test images...
Final Test Accuracy: 94.36%

Confusion Matrix:
             airplane automobile  bird  cat  deer  dog  frog horse  ship truck
airplane        ...
...
```

---

## Files

| File | Description |
|---|---|
| `cifar_evolution_engine_local.py` | Step 1 — main evolutionary search loop |
| `train_production_champion.py` | Step 2 — 200-epoch full training on best architecture |
| `evaluate_production_champion.py` | Step 3 — confusion matrix evaluation of trained model |

---

## Paper & Data

Full paper, trained model weights, and experiment database are archived on Zenodo:

**DOI: [10.5281/zenodo.20737711](https://zenodo.org/records/20737711)**

Archive includes:
- `manuscript.pdf` — full research paper
- `champion_converged_resnet.pth` — trained champion model weights (9.1 MB)
- `pow_resnet_nas.db` — full 696-generation experiment database
- `vc_whitepaper_local_ai_companion.pdf` — local AI companion whitepaper

---

## Citation

If you use this work please cite:

```bibtex
@misc{sandler2026gaar,
  author    = {Sandler, Leon},
  title     = {Privacy-Preserving Local AI Research Pipeline:
               Evolutionary NAS on CIFAR-10 using GaaR},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20737711},
  url       = {https://zenodo.org/records/20737711}
}
```

---

## arXiv Submission

This work is pending arXiv submission in **cs.NE** (Neural and Evolutionary Computing)
and **cs.CV** (Computer Vision and Pattern Recognition).

If you are an established arXiv author in either category and find this work
interesting, I would be grateful for an endorsement:

**https://arxiv.org/auth/endorse?x=H9WTH6**

— Leon Sandler
