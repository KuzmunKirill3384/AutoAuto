#!/usr/bin/env python3

from __future__ import annotations
import sys, time, argparse
from pathlib import Path
from time import sleep
from typing import Optional

import RPi.GPIO as GPIO
import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ─── Picamera2 ────────────────────────────────────────────────────────
try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("Picamera2 не установлена: sudo apt install python3-picamera2")

# ─── пины моторов ────────────────────────────────────────────────────
FRONT_LEFT_IN1, FRONT_LEFT_IN2, FRONT_LEFT_EN   = 20, 21, 16
FRONT_RIGHT_IN1, FRONT_RIGHT_IN2, FRONT_RIGHT_EN = 13, 26, 19
REAR_RIGHT_IN1, REAR_RIGHT_IN2, REAR_RIGHT_EN   = 24, 23, 25
REAR_LEFT_IN1,  REAR_LEFT_IN2,  REAR_LEFT_EN    = 7,  8,  12
PWM_FREQ_HZ = 1000
SPEED = 45        # duty-cycle ≈ 95 % — почти максимум

# ─── модель и параметры инференса ────────────────────────────────────
MODEL_PATH  = Path("robot_action_cnn.pth")
IMG_SIZE    = (96, 128)                 # H×W
ACTIONS     = ["forward","backward","rotate_left","rotate_right"]
# … (остальной код без изменений выше) …

FRAME_PERIOD = 0.01          # минимальная пауза между кадрами (~100 FPS)

# ─── основной код ───────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="Показ кадра и предсказания (нужен X11)")
    args = parser.parse_args()

    if not SCRIPT_PATH.exists():
        sys.exit("robot_action_cnn_script.pt не найден. Сначала обучите модель.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = torch.jit.load(str(SCRIPT_PATH), map_location=device)
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize(IMG_SIZE[::-1]),
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5), (.5,.5,.5)),
    ])

    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"size": (640, 480),
                                                       "format": "RGB888"}))
    cam.start(); capture = lambda: cam.capture_array()

    last = 0.0
    try:
        while True:
            now = time.time()
            if now - last < FRAME_PERIOD:
                continue
            last = now

            frame = capture()
            inp = tfms(Image.fromarray(frame)).unsqueeze(0).to(device)

            with torch.no_grad():
                act = ACTIONS[int(model(inp).argmax(1))]

            if   act == "forward":       forward()
            elif act == "backward":      backward()
            elif act == "rotate_left":   rotate_left()
            elif act == "rotate_right":  rotate_right()
            else:                        stop()

            print(act)
            if args.preview:
                import matplotlib.pyplot as plt
                plt.imshow(frame); plt.title(act); plt.pause(0.001); plt.clf()
            # убираем дополнительный sleep, чтобы не замедлять цикл
    finally:
        cam.stop()
        _all_low(); GPIO.cleanup()
        print("Exit")

if __name__ == "__main__":
    main()
