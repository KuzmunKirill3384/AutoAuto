#!/usr/bin/env python3
"""
MobileNetV3-Small trainer (macOS / M-GPU, без backward).

• dataset  : /Users/kuzminkirill/Downloads/pi
• classes  : forward / rotate_left / rotate_right
• bottom-mask 30-45 %
• полностью игнорируем backward (и stop)
• no channels_last → никакой ошибки view/stride
"""

from __future__ import annotations
import os
import random
import re
import ssl
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import numpy as np

# ─── Config ────────────────────────────────────────────────
DATASET_DIR = Path("/Users/kuzminkirill/Downloads/pi")   # путь к папке с изображениями
IMG_SIZE    = (96, 128)                                  # размер входа сети: (высота, ширина)
ACTIONS     = ["forward", "rotate_left", "rotate_right"] # целевые классы
IGNORE      = {"stop", "backward"}                       # метки, которые игнорируем

BATCH_SIZE  = 64      # число примеров в одном батче
EPOCHS      = 20      # сколько полных проходов по датасету
LR, WD      = 3e-4, 1e-4  # learning rate и weight decay для AdamW
VAL_SPLIT   = .15      # доля валидации от общего числа примеров
SEED        = 42       # сид для детерминизма

# Отключаем проверку SSL, чтобы не было ошибок при загрузке предобученных весов
ssl._create_default_https_context = ssl._create_unverified_context
torch.hub._validate_ssl = False

# регулярка ищет только нужные метки в имени файла
ACTION_RE = re.compile("|".join(ACTIONS + list(IGNORE)))

# ─── BottomMask: затемнение низа ─────────────────────────────
class BottomMask:
    """Затемняет снизу кадра черным прямоугольником."""
    def __init__(self, min_frac: float = .30, max_frac: float = .45):
        """
        min_frac, max_frac — минимальный и максимальный процент высоты
        кадра, который будет затемнен (30–45%).
        """
        self.min_frac = min_frac
        self.max_frac = max_frac

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        # выбираем случайную долю для затенения
        frac = random.uniform(self.min_frac, self.max_frac)
        y0 = int(h * (1.0 - frac))  # вычисляем старт y для прямоугольника
        img = img.copy()
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, y0), (w, h)], fill=0)
        return img

# ─── Сборка списка файлов ➜ фильтруем backward и stop ───────────
def collect_samples() -> list[tuple[Path,int]]:
    files = list(DATASET_DIR.rglob("*.[jp][pn]g"))  # ищем jpg и png
    if not files:
        sys.exit(f"No images in {DATASET_DIR}")
    out = []
    for f in files:
        m = ACTION_RE.search(f.stem)
        if not m:
            continue
        lbl = m.group()
        if lbl in IGNORE:  # пропускаем ненужные метки
            continue
        out.append((f, ACTIONS.index(lbl)))
    random.shuffle(out)
    return out

# ─── Функция разбивки на train/val ────────────────────────────────────
def split(arr: list[tuple[Path,int]], ratio: float) -> tuple[list, list]:
    """
    ratio — доля данных ушедших на валидацию (например, .15 = 15%).
    Возвращает два списка: train и val.
    """
    per = defaultdict(list)
    for p, l in arr:
        per[l].append((p, l))
    tr, va = [], []
    for l, lst in per.items():
        random.shuffle(lst)
        k = int(len(lst) * ratio)
        va += lst[:k]
        tr += lst[k:]
    random.shuffle(tr)
    random.shuffle(va)
    return tr, va

# ─── Dataset-класс для PyTorch ───────────────────────────────────────
class RobotDS(Dataset):
    def __init__(self, items: list[tuple[Path,int]], tfm: transforms.Compose):
        self.items = items  # список (путь, метка)
        self.tfm   = tfm    # трансформации

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        p, lbl = self.items[idx]
        img = Image.open(p).convert("RGB")
        return self.tfm(img), lbl

# ─── Создание модели ────────────────────────────────────────────────
def build_model() -> nn.Module:
    """
    Загружаем MobileNetV3-Small с предобученными весами ImageNet,
    меняем последний слой под число наших классов.
    """
    net = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    in_f = net.classifier[3].in_features
    net.classifier[3] = nn.Linear(in_f, len(ACTIONS))
    return net

# ─── Основной тренировочный цикл ────────────────────────────────────
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # 1. Собираем и разделяем данные
    samples      = collect_samples()
    train_it, val_it = split(samples, VAL_SPLIT)
    print(f"train {len(train_it)} | val {len(val_it)}")

    # 2. Определяем трансформации
    aug = transforms.Compose([
        BottomMask(.30, .45),  # затемняем низ кадра
        transforms.Resize((IMG_SIZE[0] + 24, IMG_SIZE[1] + 24),
                          interpolation=InterpolationMode.BILINEAR),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.3, .3, .3, .1),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    ])
    plain = transforms.Compose([
        BottomMask(.30, .45),
        transforms.Resize(IMG_SIZE[::-1]),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    ])

    # 3. Сложная часть: создаем взвешенный семплер для балансировки классов
    hist = np.bincount([lbl for _, lbl in train_it], minlength=len(ACTIONS))
    w = 1.0 / torch.tensor(hist, dtype=torch.float32)
    sampler = WeightedRandomSampler(
        [w[l].item() for _, l in train_it],
        num_samples=len(train_it),
        replacement=True
    )

    # 4. Создаем загрузчики данных
    tr_dl = DataLoader(
        RobotDS(train_it, aug),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=os.cpu_count(),
        persistent_workers=True,
        prefetch_factor=4
    )
    va_dl = DataLoader(
        RobotDS(val_it, plain),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
        persistent_workers=True,
        prefetch_factor=4
    )

    # 5. Выбираем устройство (MPS на Mac или CPU)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", dev)

    # 6. Строим модель и замораживаем backbone на первые 3 эпохи
    net = build_model().to(dev)
    for p in net.features.parameters():
        p.requires_grad = False

    # 7. Настраиваем оптимизатор и lr-расписание
    opt   = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=LR * 10,
        total_steps=EPOCHS * len(tr_dl)
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    for ep in range(1, EPOCHS + 1):
        # разморозка backbone после 3 эпох
        if ep == 4:
            for p in net.features.parameters():
                p.requires_grad = True

        # --- фаза обучения ---
        net.train()
        for x, y in tqdm(tr_dl, desc=f"Epoch {ep}/{EPOCHS}", leave=False):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(net(x), y)  # считаем лосс
            loss.backward()
            opt.step()
            sched.step()

        # --- фаза валидации ---
        net.eval()
        correct = 0
        with torch.no_grad():
            for x, y in va_dl:
                pred = net(x.to(dev)).argmax(1).cpu()
                correct += (pred == y).sum().item()
        acc = 100 * correct / len(val_it)
        print(f"Epoch {ep}: val acc {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), "robot_action_cnn.pth")
            print("✓ saved best model")

    # 8. Экспорт пропускной модели в TorchScript
    torch.jit.script(net.cpu()).save("robot_action_cnn_script.pt")
    print("Training done, best acc:", best_acc)

if __name__ == "__main__":
    main()
