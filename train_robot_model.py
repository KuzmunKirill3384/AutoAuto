#!/usr/bin/env python3
"""
Tiny-CNN trainer for four-wheel robot actions **с автоматической очисткой "стоп"-кадров**.

• Перед обучением на лету фильтрует и *удаляет* кадры, записанные в состоянии stop.
  – Строки с action == "stop" вырезаются из labels.csv.
  – Соответствующие изображения (по-умолчанию) удаляются с диска.
  – Создаёт резервную копию оригинального labels.csv (labels.csv.bak).
• Далее обучает лёгкую CNN на четырёх оставшихся действиях.
• Показывает tqdm-прогресс и (опционально) зажигает LED на GPIO18.
• Сохраняет модель в robot_action_cnn.pth.

Запуск:
    python3 train_robot_model.py
Никакой дополнительной настройки не требуется.
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# ────────────────────────── GPIO (опц.) ──────────────────────────
try:
    import RPi.GPIO as GPIO  # type: ignore
except (ImportError, RuntimeError):
    GPIO = None  # не на Pi

# ────────────────────────── ML ─────────────────────────────———
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
    from torchvision import transforms
except ImportError:
    sys.stderr.write(
        "❌  PyTorch not found. Install with:\n"
        "    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu\n"
    )
    raise

from PIL import Image  # type: ignore
from tqdm import tqdm  # type: ignore

# ────────────────────────── config ───────────────────────────
DATASET_DIR = Path("dataset")
LABELS_CSV = DATASET_DIR / "labels.csv"
LED_PIN = 18               # None → отключить LED
IMG_SIZE = (96, 128)       # H×W
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
VAL_SPLIT = 0.15
CLEAN_STOP_IMAGES = True   # удалять jpg кадров «stop»
ACTIONS = [
    "forward",
    "backward",
    "rotate_left",
    "rotate_right",
]

# ────────────────────────── helpers ──────────────────────────

def _prepare_led():
    if GPIO is None or LED_PIN is None:
        return None
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT)
    return LED_PIN

# ---------- DATASET CLEANUP ----------------------------------

def clean_dataset():
    """Удаляет записи/файлы с action==stop. Создаёт labels.csv.bak."""
    if not LABELS_CSV.exists():
        sys.exit(f"❌  {LABELS_CSV} not found – нечего чистить")

    backup = LABELS_CSV.with_suffix(".csv.bak")
    if not backup.exists():
        shutil.copy(LABELS_CSV, backup)

    kept_lines: List[str] = []
    removed = 0
    with backup.open() as fp:
        reader = csv.reader(fp)
        for row in reader:
            fname, action, *_ = row
            if action == "stop":
                removed += 1
                if CLEAN_STOP_IMAGES:
                    img_path = DATASET_DIR / fname
                    try:
                        img_path.unlink(missing_ok=True)
                    except Exception as e:
                        print(f"⚠️  Could not delete {img_path}: {e}")
                continue
            kept_lines.append(",".join(row))

    LABELS_CSV.write_text("\n".join(kept_lines) + "\n")
    print(f"🧹  Cleaned dataset: removed {removed} 'stop' samples, kept {len(kept_lines)} others")

class RobotDataset(Dataset):
    def __init__(self, root: Path, labels_file: Path, tfms: transforms.Compose):
        self.samples: List[Tuple[Path, int]] = []
        self.tfms = tfms
        with labels_file.open() as fp:
            reader = csv.reader(fp)
            for row in reader:
                fname, action, *_ = row
                if action not in ACTIONS:
                    continue
                img_path = root / fname
                if img_path.exists():
                    self.samples.append((img_path, ACTIONS.index(action)))
        if not self.samples:
            raise RuntimeError("No training samples found – проверьте dataset после очистки")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.tfms(img), label

class TinyCNN(nn.Module):
    def __init__(self, n_classes=len(ACTIONS)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        dummy = torch.zeros(1, 3, *IMG_SIZE)
        with torch.no_grad():
            n_flat = self._conv_forward(dummy).numel()
        self.fc1 = nn.Linear(n_flat, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def _conv_forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._conv_forward(x)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ────────────────────────── training ─────────────────────────

def train():
    if not DATASET_DIR.exists():
        sys.exit(f"❌  {DATASET_DIR}/ not found – положите туда данные")

    clean_dataset()

    tfms = transforms.Compose([
        transforms.Resize(IMG_SIZE[::-1]),
        transforms.ToTensor(),
    ])
    full_ds = RobotDataset(DATASET_DIR, LABELS_CSV, tfms)

    val_len = int(len(full_ds) * VAL_SPLIT)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    loader = lambda d, shuf: DataLoader(d, BATCH_SIZE, shuffle=shuf, num_workers=2, pin_memory=False)
    train_loader, val_loader = loader(train_ds, True), loader(val_ds, False)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    led = _prepare_led()
    if led is not None:
        GPIO.output(led, GPIO.HIGH)

    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
            for imgs, lbls in pbar:
                imgs, lbls = imgs.to(dev), lbls.to(dev)
                opt.zero_grad()
                loss = crit(model(imgs), lbls)
                loss.backward(); opt.step()
                total_loss += loss.item() * imgs.size(0)
                pbar.set_postfix(loss=total_loss / ((pbar.n + 1) * BATCH_SIZE))

            model.eval(); correct = total = 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    out = model(imgs.to(dev))
                    pred = out.argmax(1)
                    total += lbls.size(0)
                    correct += (pred.cpu() == lbls).sum().item()
            acc = 100 * correct / total
            print(f"\n✅  Val accuracy: {acc:.2f}%  ({correct}/{total})\n")

        torch.save(model.state_dict(), "robot_action_cnn.pth")
        print("🎉  Done. Model saved → robot_action_cnn.pth")
    finally:
        if led is not None:
            GPIO.output(led, GPIO.LOW); GPIO.cleanup(led)

if __name__ == "__main__":
    train()
