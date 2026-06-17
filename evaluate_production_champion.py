#!/usr/bin/env python3
"""
Evolutionary AI Engine - Production Model Evaluation Suite
Loads compiled champion weights, runs against the test set, and generates metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_CHECKPOINT_PATH = "./champion_converged_resnet.pth"
CHAMPION_BASE_CH = 32   # Matches your Step 4 Champion Architecture
CHAMPION_NUM_BLOCKS = 4 # Matches your Step 4 Champion Architecture
CHAMPION_USE_SE = True  # Matches your Step 4 Champion Architecture

CIFAR10_CLASSES = [
    'plane', 'car', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Device Configuration for DirectML / Windows AMD GPU processing
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch, "dml") or "dml" in str(torch.__dict__):
    device = torch.device("dml")
else:
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device("cpu")

# ── 1. Model Definitions (Must match phenotype blueprint exactly) ────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(),
            nn.Linear(reduced, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.fc(x).view(b, c, 1, 1)
        return x * w

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut_val = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.se(out)
        return out + shortcut_val

class ResNetPhenotype(nn.Module):
    def __init__(self, base_ch=32, num_blocks=2, use_se=True):
        super().__init__()
        self.in_channels = base_ch
        self.prep = nn.Conv2d(3, base_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_group(base_ch, num_blocks, stride=1, use_se=use_se)
        self.layer2 = self._make_group(base_ch * 2, num_blocks, stride=2, use_se=use_se)
        self.layer3 = self._make_group(base_ch * 4, num_blocks, stride=2, use_se=use_se)
        self.bn = nn.BatchNorm2d(base_ch * 4)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base_ch * 4, 10)

    def _make_group(self, channels, num_blocks, stride, use_se):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, channels, s, use_se))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.prep(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(self.relu(self.bn(out)))
        out = out.view(out.size(0), -1)
        return self.classifier(out)

# ── 2. Run Evaluation Pipeline ───────────────────────────────────────────────
def run_evaluation():
    print(f"🖥️  Loading Compute Target Backend: {device}")
    
    # 1. Verification Checklist for the Model Weights File
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"❌ Error: Model weights file '{MODEL_CHECKPOINT_PATH}' not found yet.")
        print("   Wait for 'train_production_champion.py' to finish compiling weights first.")
        sys.exit(1)
        
    print(f"📥 Loading untouched CIFAR-10 test partition...")
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    
    # 2. Re-instantiate Champion Network and Map Saved Weights
    print("🧬 Instantiating structural phenotype network parameters...")
    model = ResNetPhenotype(
        base_ch=CHAMPION_BASE_CH,
        num_blocks=CHAMPION_NUM_BLOCKS,
        use_se=CHAMPION_USE_SE
    ).to(device)
    
    print("💾 Mapping learned parameters onto model graph...")
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
    model.eval()
    
    # Initialize matrix trackers (10x10 zero grid)
    confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)
    correct, total = 0, 0
    
    print("⏳ Evaluating target datasets across verification batch distributions...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            
            correct += preds.eq(target).sum().item()
            total += data.size(0)
            
            # Map predictions to confusion matrix rows and columns
            for t, p in zip(target.view(-1), preds.view(-1)):
                confusion_matrix[t.item(), p.item()] += 1
                
    final_test_acc = (correct / total) * 100.0
    
    # ── 3. Print Performance Report Outputs ──────────────────────────────────
    print("\n" + "="*80)
    print(f"🏁  FINAL ARXIV PRODUCTION BENCHMARK REPORT")
    print("="*80)
    print(f"✅ Absolute Global Accuracy on Untouched Test Set: {final_test_acc:.2f}%\n")
    
    print("📊 TEXT-BASED CONFUSION MATRIX REPRESENTATION:")
    print("   Horizontal columns = Predicted Class | Vertical rows = Actual Ground Truth\n")
    
    # Header format string
    # FIX: Remove the backslash from inside the f-string curly braces
    label_text = "True \\ Pred"
    header_str = f"{label_text:<12}" + "".join([f"{cls:>7}" for cls in CIFAR10_CLASSES])

    print(header_str)
    print("-" * len(header_str))
    
    for idx, row in enumerate(confusion_matrix):
        row_str = f"{CIFAR10_CLASSES[idx]:<12}" + "".join([f"{val.item():>7}" for val in row])
        print(row_str)
        
    print("\n🎯 INDIVIDUAL CLASS PRECISION PERFORMANCE METRICS:")
    print("-" * 50)
    for idx in range(10):
        actual_total = confusion_matrix[idx, :].sum().item()
        predicted_correct = confusion_matrix[idx, idx].item()
        class_acc = (predicted_correct / actual_total) * 100.0 if actual_total > 0 else 0.0
        print(f" • Class {CIFAR10_CLASSES[idx]:<7} Generalization Matrix Score: {class_acc:.2f}%")
    print("="*80)

if __name__ == "__main__":
    run_evaluation()
