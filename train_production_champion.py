#!/usr/bin/env python3
"""
Evolutionary AI Engine - Standalone Production Training Pipeline
Extracts the Rank 1 Leaderboard Champion and Trains to Full Convergence (200 Epochs)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import sqlite3
import random
import os
import sys
import logging
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = "./pow_resnet_nas.db"
PRODUCTION_EPOCHS = 200
MODEL_CHECKPOINT_PATH = "./champion_converged_resnet.pth"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

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

log.info(f"🖥️ Production Compute Target Locked: {device}")

# ── 1. Extract Champion Genome from Leaderboard ───────────────────────────────
def extract_champion():
    if not os.path.exists(DB_PATH):
        log.error(f"❌ Database file '{DB_PATH}' not found! Run your search engine first.")
        sys.exit(1)
        
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT lr, batch_size, base_ch, num_blocks, use_se, weight_decay, mixup_alpha, score, total_hashes "
        "FROM experiments ORDER BY score DESC LIMIT 1"
    ).fetchone()
    conn.close()
    
    if not row:
        log.error("❌ No evolutionary runs found in the database leaderboard yet.")
        sys.exit(1)
        
    champion = {
        "lr": row[0], "batch_size": int(row[1]), "base_ch": int(row[2]),
        "num_blocks": int(row[3]), "use_se": row[4], "weight_decay": row[5],
        "mixup_alpha": row[6], "search_score": row[7], "generation": row[8]
    }
    return champion

champ = extract_champion()
log.info(f"🧬 CHAMPION GENOME RETRIEVED (From Search Generation {champ['generation']}):")
log.info(f"   • Baseline Search Validation Score: {champ['search_score']:.2f}%")
log.info(f"   • Topology: [Channels: {champ['base_ch']} | Blocks: {champ['num_blocks']} | SE: {'On' if champ['use_se'] >= 0.5 else 'Off'}]")
log.info(f"   • Hyperparameters: [LR: {champ['lr']:.4f} | Batch: {champ['batch_size']} | WD: {champ['weight_decay']:.5e} | Mixup: {champ['mixup_alpha']}]")

# ── 2. Full-Scale Production Production Data Augmentations ────────────────────
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

raw_train_set = datasets.CIFAR10("./data", train=True, download=True)

torch.manual_seed(42)
train_size = int(0.9 * len(raw_train_set))
val_size = len(raw_train_set) - train_size
raw_train, raw_val = random_split(raw_train_set, [train_size, val_size])

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        return self.transform(x), y
    def __len__(self):
        return len(self.subset)

train_loader = DataLoader(TransformedDataset(raw_train, train_transform), batch_size=champ["batch_size"], shuffle=True)
val_loader = DataLoader(TransformedDataset(raw_val, val_transform), batch_size=champ["batch_size"], shuffle=False)

# ── 3. Production Architecture Blocks ─────────────────────────────────────────
def apply_mixup(data, targets, alpha):
    if alpha > 0:
        lam = random.betavariate(alpha, alpha)
    else:
        lam = 1.0
    batch_size = data.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_data = lam * data + (1 - lam) * data[index, :]
    targets_a, targets_b = targets, targets[index]
    return mixed_data, targets_a, targets_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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

# ── 4. Full Convergence Engine Execution ──────────────────────────────────────
def run_production():
    model = ResNetPhenotype(
        base_ch=champ["base_ch"],
        num_blocks=champ["num_blocks"],
        use_se=(champ["use_se"] >= 0.5)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=champ["lr"], momentum=0.9, weight_decay=champ["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRODUCTION_EPOCHS * len(train_loader))
    
    best_val_acc = 0.0
    log.info(f"🚀 Starting Full Standalone Train Lifecycle ({PRODUCTION_EPOCHS} Epochs)...")
    
    for epoch in range(1, PRODUCTION_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data, target_a, target_b, lam = apply_mixup(data, target, champ["mixup_alpha"])
            
            optimizer.zero_grad()
            output = model(data)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
        # Complete validation step per epoch
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
                
        val_acc = (correct / total) * 100.0
        epoch_loss = running_loss / len(train_loader)
        
        # Track and save all-time best state parameters
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
            save_status = "⭐ NEW BEST"
        else:
            save_status = ""
            
        log.info(f"Epoch [{epoch:03d}/{PRODUCTION_EPOCHS}] Loss: {epoch_loss:.4f} | Val Accuracy: {val_acc:.2f}% {save_status}")
        
    log.info("=" * 75)
    log.info(f"🏁 Production Convergence Complete!")
    log.info(f"🏆 Peak Publishable Validation Accuracy: {best_val_acc:.2f}%")
    log.info(f"💾 Elite model weights safely compiled inside: {MODEL_CHECKPOINT_PATH}")

if __name__ == "__main__":
    run_production()
