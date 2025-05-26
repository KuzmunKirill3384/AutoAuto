#!/usr/bin/env python3
"""
Autonomous 4-wheel robot (Picamera2)

• Модель MobileNetV3-Small с 3 выходами: forward, rotate_left, rotate_right
• Каждые 0.01 с берёт кадр 640×480 → 128×96, нормализует,
  предсказывает действие и подаёт команды на GPIO.

Файл весов должен быть обучён именно на 3-классовую задачу!
"""

from __future__ import annotations
import sys, time, argparse
from pathlib import Path

import RPi.GPIO as GPIO
import torch, torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ─── Picamera2 ─────────────────────────────────────────────
try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("sudo apt install python3-picamera2")

# ─── GPIO pin-map (как в рабочем 4-классовом скрипте) ─────
FRONT_LEFT_IN1,  FRONT_LEFT_IN2,  FRONT_LEFT_EN  = 20, 21, 16
FRONT_RIGHT_IN1, FRONT_RIGHT_IN2, FRONT_RIGHT_EN = 13, 26, 19
REAR_RIGHT_IN1,  REAR_RIGHT_IN2,  REAR_RIGHT_EN  = 24, 23, 25
REAR_LEFT_IN1,   REAR_LEFT_IN2,   REAR_LEFT_EN   = 7,  8,  12

PWM_FREQ_HZ = 1000   # Гц
SPEED       = 20    # % duty-cycle

# ─── модель и инференс ────────────────────────────────────
MODEL_PATH   = Path("robot_action_cnn_2.pth")   # веса 3-классовой модели
IMG_SIZE     = (96, 128)            # H×W
ACTIONS      = ["forward", "rotate_left", "rotate_right"]
FRAME_PERIOD = 0.01                 # ≈100 FPS

# ─── GPIO init ────────────────────────────────────────────
GPIO.setmode(GPIO.BCM); GPIO.setwarnings(False)
def _setup_motor(in1, in2, en):
    GPIO.setup(in1, GPIO.OUT); GPIO.setup(in2, GPIO.OUT); GPIO.setup(en, GPIO.OUT)
    GPIO.output(in1, GPIO.LOW); GPIO.output(in2, GPIO.LOW)
    pwm = GPIO.PWM(en, PWM_FREQ_HZ); pwm.start(SPEED)
    return pwm

# инициализируем все 4 драйвера
front_left_pwm  = _setup_motor(FRONT_LEFT_IN1,  FRONT_LEFT_IN2,  FRONT_LEFT_EN)
front_right_pwm = _setup_motor(FRONT_RIGHT_IN1, FRONT_RIGHT_IN2, FRONT_RIGHT_EN)
rear_left_pwm   = _setup_motor(REAR_LEFT_IN1,   REAR_LEFT_IN2,   REAR_LEFT_EN)
rear_right_pwm  = _setup_motor(REAR_RIGHT_IN1,  REAR_RIGHT_IN2,  REAR_RIGHT_EN)


# ─── движения (идентичны «рабочему» скрипту) ─────────────
def _all_low():
    for pin in (
        FRONT_LEFT_IN1,  FRONT_LEFT_IN2,
        FRONT_RIGHT_IN1, FRONT_RIGHT_IN2,
        REAR_LEFT_IN1,   REAR_LEFT_IN2,
        REAR_RIGHT_IN1,  REAR_RIGHT_IN2
    ):
        GPIO.output(pin, GPIO.LOW)

def stop():         _all_low()

def forward():
    _all_low()
    GPIO.output(FRONT_LEFT_IN1,  GPIO.HIGH); GPIO.output(REAR_LEFT_IN1,  GPIO.HIGH)
    GPIO.output(FRONT_RIGHT_IN1, GPIO.HIGH); GPIO.output(REAR_RIGHT_IN1, GPIO.HIGH)

def rotate_left():
    _all_low()
    GPIO.output(FRONT_RIGHT_IN1, GPIO.HIGH); GPIO.output(REAR_RIGHT_IN1, GPIO.HIGH)
    GPIO.output(FRONT_LEFT_IN2,  GPIO.HIGH); GPIO.output(REAR_LEFT_IN2,  GPIO.HIGH)

def rotate_right():
    _all_low()
    GPIO.output(FRONT_LEFT_IN1,  GPIO.HIGH); GPIO.output(REAR_LEFT_IN1,  GPIO.HIGH)
    GPIO.output(FRONT_RIGHT_IN2, GPIO.HIGH); GPIO.output(REAR_RIGHT_IN2, GPIO.HIGH)

# ─── модель ───────────────────────────────────────────────
def build_model():
    net = models.mobilenet_v3_small(weights=None, width_mult=1.0)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(ACTIONS))
    return net

# ─── MAIN ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="Показ кадра (нужен X11)")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        sys.exit(f"{MODEL_PATH} не найден – обучите/скопируйте веса для 3 классов.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize(IMG_SIZE[::-1]),   # (W, H)
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5), (.5,.5,.5)),
    ])

    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}))
    cam.start();  capture = lambda: cam.capture_array()

    last = 0.0
    try:
        while True:
            now = time.time()
            if now - last < FRAME_PERIOD:
                continue
            last = now

            frame = capture()
            inp   = tfms(Image.fromarray(frame)).unsqueeze(0).to(device)

            with torch.no_grad():
                action = ACTIONS[int(model(inp).argmax(1))]

            {
                "forward":      forward,
                "rotate_left":  rotate_left,
                "rotate_right": rotate_right
            }.get(action, stop)()

            print(action)
            if args.preview:
                import matplotlib.pyplot as plt
                plt.imshow(frame); plt.title(action); plt.pause(0.001); plt.clf()
    finally:
        cam.stop()
        _all_low()
        GPIO.cleanup()
        print("Exit")

if __name__ == "__main__":
    main()
