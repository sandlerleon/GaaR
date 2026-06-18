# Privacy-Preserving Local AI Research Pipeline: Evolutionary NAS on CIFAR-10 using GaaR

Bio-inspired NAS engine using GaaR Network, Banana Core homeostasis, 
and StruggleLayer adaptive hardening.

## Files
- cifar_evolution_engine_local.py — main evolutionary search loop
- train_production_champion.py — 200-epoch production training
- evaluate_production_champion.py — confusion matrix evaluation

## Paper
Published on Zenodo: [https://zenodo.org/records/20737711]

This archive contains the source code, trained model, experiment database, and documentation for a privacy-preserving evolutionary Neural Architecture Search (NAS) framework that runs entirely on local consumer hardware. The system combines three bio-inspired control mechanisms—Banana Core homeostasis, StruggleLayer adaptive hardening, and the GaaR (Goal-Aware Adaptive Regulation) Network—to perform efficient architecture search under noisy, low-cost evaluation conditions.

The framework supports AMD GPUs through DirectML, Apple Silicon through Metal Performance Shaders (MPS), NVIDIA CUDA, and CPU execution, enabling zero-data-egress AI research without cloud infrastructure. Local AI assistants such as Ollama, LM Studio, and Claude Cowork can be integrated through a SQLite experiment database to provide privacy-preserving experiment analysis and mutation recommendations.

In a 696-generation CIFAR-10 search conducted on a single AMD Radeon RX 5700 GPU, the system discovered a champion architecture achieving 94.36% test accuracy after full training. The archive includes source code, trained weights, experiment logs, and supporting documentation to facilitate reproducibility and further research into local AI and privacy-preserving AutoML.

## Accuracy Claim as targeted towards Apple's silicon M5 AI chips present in Mac Minis PCs and MacBooks Laptops  

The manuscript now consistently reports:

**696** generations
**45 minutes** search time
**94.36%** CIFAR-10 test accuracy (currently still below 99.9% of cloud based AI but sufficient enough)

