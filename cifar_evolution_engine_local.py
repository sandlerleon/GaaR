#!/usr/bin/env python3
"""
Evolutionary AI Engine - CIFAR-10 Residual NAS with Attention & Mixup
Designed for DirectML/AMD Backends to achieve publication-grade (>95%+) accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import sqlite3
import math
import random
import os
import logging
import smtplib
import time
from datetime import datetime
from email.mime.text import MIMEText

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = os.environ.get("POW_DB", "./pow_resnet_nas.db")
BITCOIN_TARGET = 1e5
NOTIFY_EMAIL = "eliptum@gmail.com"
AI_MILESTONES = [70.0, 80.0, 85.0, 90.0, 93.0, 95.0, 97.0]

# Advanced, ArXiv-Grade Publishable Genome Configuration
BANANA_CORE = {
    "learning_rate": 0.05,
    "batch_size": 128,
    "base_channels": 32,      # Multiplied up the residual ladder (32 -> 64 -> 128)
    "num_blocks": 2,          # Number of pre-activation residual blocks per group
    "use_se": 1.0,            # 1.0 = Enable Squeeze-and-Excitation blocks, 0.0 = Off
    "weight_decay": 5e-4,     # Evolved weight decay factor
    "mixup_alpha": 0.2,       # Mixup data augmentation distribution coefficient
}
BANANA_RATE = 0.50

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

log.info(f"🖥️ Target Compute locked on hardware backend: {device}")

# ── 1. Advanced Publication Data Pipeline ──────────────────────────────────────
log.info("📥 Constructing Double-Buffered RAM Dataset Buffers...")

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

GLOBAL_TRAIN_SET = TransformedDataset(raw_train, train_transform)
GLOBAL_VAL_SET = TransformedDataset(raw_val, val_transform)

# ── 2. Mixup Augmentation Execution Logic ────────────────────────────────────
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

# ── 3. Advanced Squeeze-and-Excitation Residual Architecture ─────────────────
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
        # FIX: Corrected array replication syntax
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

# ── 4. Database Sync System ───────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, approach TEXT, 
            lr REAL, batch_size INT, base_ch INT, num_blocks INT, use_se REAL, 
            weight_decay REAL, mixup_alpha REAL, score REAL, total_hashes INT
        )
    """)
    conn.execute("CREATE TABLE IF NOT EXISTS emails (threshold REAL PRIMARY KEY, sent_at TEXT)")
    conn.commit()
    return conn

def get_best(conn):
    row = conn.execute(
        "SELECT lr, batch_size, base_ch, num_blocks, use_se, weight_decay, mixup_alpha, score FROM experiments "
        "ORDER BY score DESC LIMIT 1"
    ).fetchone()
    if row:
        return {
            "learning_rate": row[0], "batch_size": int(row[1]), "base_channels": int(row[2]),
            "num_blocks": int(row[3]), "use_se": row[4], "weight_decay": row[5], "mixup_alpha": row[6]
        }
    return dict(BANANA_CORE)

# ── 5. Bio-Inspired Evolutionary Operators ────────────────────────────────────
def mutate_genome(params):
    mutated = dict(params)
    if random.random() < BANANA_RATE:
        mutated["learning_rate"] = max(1e-3, min(0.2, mutated["learning_rate"] * random.choice([0.85, 1.15])))
        mutated["batch_size"] = int(random.choice([32, 64, 128]))
        mutated["weight_decay"] = max(1e-5, min(1e-2, mutated["weight_decay"] * random.choice([0.9, 1.1])))
        mutated["mixup_alpha"] = max(0.0, min(1.0, mutated["mixup_alpha"] + random.choice([-0.05, 0.05])))
        
        mutated["base_channels"] = int(max(16, min(64, mutated["base_channels"] + random.choice([-8, 8]))))
        mutated["num_blocks"] = int(max(1, min(4, mutated["num_blocks"] + random.choice([-1, 1]))))
        mutated["use_se"] = 1.0 if random.random() > 0.3 else 0.0
    return mutated

# ── 6. ArXiv Candidate Evaluation Loop (5 Full Training Epochs) ───────────────
def train_and_evaluate_candidate(params):
    try:
        train_loader = torch.utils.data.DataLoader(GLOBAL_TRAIN_SET, batch_size=params["batch_size"], shuffle=True)
        val_loader = torch.utils.data.DataLoader(GLOBAL_VAL_SET, batch_size=params["batch_size"], shuffle=False)
        
        model = ResNetPhenotype(
            base_ch=int(params["base_channels"]),
            num_blocks=int(params["num_blocks"]),
            use_se=(params["use_se"] >= 0.5)
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"], momentum=0.9, weight_decay=params["weight_decay"])
        
        epochs = 5
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
        
        for epoch in range(epochs):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                data, target_a, target_b, lam = apply_mixup(data, target, params["mixup_alpha"])
                
                optimizer.zero_grad()
                output = model(data)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)

                loss.backward()
                optimizer.step()
                scheduler.step()

        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

        return (correct / total) * 100.0

    except Exception as e:
        log.error(f"Execution boundary fault bypassed safely: {e}")
        return 0.0

# ── 7. Infinite Evolutionary Orchestrator Loop ──────────────────────────────────
def run_loop():
    conn = init_db()

    total_generations = 0
    last_best_score = 0.0

    log.info("🚀 Continuous Search space migrated to Residual-Attention macro structures...")

    try:
        while True:
            total_generations += 1

            parent_params = get_best(conn) if total_generations > 1 else dict(BANANA_CORE)
            active_params = mutate_genome(parent_params)

            score = train_and_evaluate_candidate(active_params)

            approach_str = "exploit_best" if score > last_best_score else "explore_mutation"

            if score > last_best_score:
                last_best_score = score

            conn.execute(
                """INSERT INTO experiments
                (timestamp, approach, lr, batch_size, base_ch, num_blocks,
                 use_se, weight_decay, mixup_alpha, score, total_hashes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    approach_str,
                    active_params["learning_rate"],
                    active_params["batch_size"],
                    active_params["base_channels"],
                    active_params["num_blocks"],
                    active_params["use_se"],
                    active_params["weight_decay"],
                    active_params["mixup_alpha"],
                    score,
                    total_generations,
                ),
            )
            conn.commit()

            se_flag = "SE=On" if active_params["use_se"] >= 0.5 else "SE=Off"

            log.info(
                f"[{approach_str}] Step={total_generations} "
                f"Topology=[Ch:{active_params['base_channels']}|"
                f"Blk:{active_params['num_blocks']}|{se_flag}] "
                f"Validation Acc={score:.2f}%"
            )

            time.sleep(1)

    except KeyboardInterrupt:
        log.info("\n🛑 Active search safely halted. Leaderboard genomes saved intact.")


if __name__ == "__main__":
    run_loop()
