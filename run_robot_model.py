#!/usr/bin/env python3
"""
Autonomous 4-wheel robot (Picamera2).

• Загружает state-dict MobileNetV3-Small из robot_action_cnn.pth.
• Каждые 0.01 с берёт кадр 640×480, ресайзит до 128×96, нормализует,
  прогнозирует одно из четырёх действий и даёт команду моторам по GPIO.
"""

from __future__ import annotations
import sys, time, argparse
from pathlib import Path
from time import sleep

import RPi.GPIO as GPIO
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ─── Picamera2 ──────────────────────────────────────────────
try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("Picamera2 не установлена: sudo apt install python3-picamera2")

# ─── пины моторов ───────────────────────────────────────────
FRONT_LEFT_IN1, FRONT_LEFT_IN2, FRONT_LEFT_EN   = 20, 21, 16
FRONT_RIGHT_IN1, FRONT_RIGHT_IN2, FRONT_RIGHT_EN = 13, 26, 19
REAR_RIGHT_IN1, REAR_RIGHT_IN2, REAR_RIGHT_EN   = 24, 23, 25
REAR_LEFT_IN1,  REAR_LEFT_IN2,  REAR_LEFT_EN    = 7,  8,  12
PWM_FREQ_HZ = 1000
SPEED       = 30           # duty-cycle %

# ─── модель и параметры инференса ───────────────────────────
MODEL_PATH   = Path("robot_action_cnn.pth")
IMG_SIZE     = (96, 128)     # H×W
ACTIONS      = ["forward", "backward", "rotate_left", "rotate_right"]
FRAME_PERIOD = 0.01          # минимальная пауза между кадрами

# ─── GPIO init ──────────────────────────────────────────────
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
def _setup_motor(in1, in2, en):
    GPIO.setup(in1, GPIO.OUT); GPIO.setup(in2, GPIO.OUT); GPIO.setup(en, GPIO.OUT)
    GPIO.output(in1, GPIO.LOW); GPIO.output(in2, GPIO.LOW)
    pwm = GPIO.PWM(en, PWM_FREQ_HZ); pwm.start(SPEED)
    return pwm

front_left_pwm  = _setup_motor(FRONT_LEFT_IN1,  FRONT_LEFT_IN2,  FRONT_LEFT_EN)
front_right_pwm = _setup_motor(FRONT_RIGHT_IN1, FRONT_RIGHT_IN2, FRONT_RIGHT_EN)
rear_left_pwm   = _setup_motor(REAR_LEFT_IN1,   REAR_LEFT_IN2,   REAR_LEFT_EN)
rear_right_pwm  = _setup_motor(REAR_RIGHT_IN1,  REAR_RIGHT_IN2,  REAR_RIGHT_EN)

# ─── функции движения ──────────────────────────────────────
def _all_low():
    for pin in (
        FRONT_LEFT_IN1, FRONT_LEFT_IN2,
        FRONT_RIGHT_IN1, FRONT_RIGHT_IN2,
        REAR_LEFT_IN1, REAR_LEFT_IN2,
        REAR_RIGHT_IN1, REAR_RIGHT_IN2
    ):
        GPIO.output(pin, GPIO.LOW)

def stop():         _all_low()
def forward():      _all_low(); GPIO.output(FRONT_LEFT_IN1,GPIO.HIGH); GPIO.output(REAR_LEFT_IN1,GPIO.HIGH); GPIO.output(FRONT_RIGHT_IN1,GPIO.HIGH); GPIO.output(REAR_RIGHT_IN1,GPIO.HIGH)
def backward():     _all_low(); GPIO.output(FRONT_LEFT_IN2,GPIO.HIGH); GPIO.output(REAR_LEFT_IN2,GPIO.HIGH); GPIO.output(FRONT_RIGHT_IN2,GPIO.HIGH); GPIO.output(REAR_RIGHT_IN2,GPIO.HIGH)
def rotate_left():  _all_low(); GPIO.output(FRONT_RIGHT_IN1,GPIO.HIGH); GPIO.output(REAR_RIGHT_IN1,GPIO.HIGH); GPIO.output(FRONT_LEFT_IN2,GPIO.HIGH); GPIO.output(REAR_LEFT_IN2,GPIO.HIGH)
def rotate_right(): _all_low(); GPIO.output(FRONT_LEFT_IN1,GPIO.HIGH); GPIO.output(REAR_LEFT_IN1,GPIO.HIGH); GPIO.output(FRONT_RIGHT_IN2,GPIO.HIGH); GPIO.output(REAR_RIGHT_IN2,GPIO.HIGH)

# ─── BUILD TINY MODEL (MobileNetV3-Small) ───────────────────
def build_model(num_classes=4):
    # подбираем width_mult = 1.0, т.к. мы сохраняли state_dict именно под эту конфигурацию
    net = models.mobilenet_v3_small(weights=None, width_mult=1.0)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
    return net

# ─── ОСНОВНОЙ КОД ───────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="Показ кадра и предсказания (нужен X11)")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        sys.exit(f"{MODEL_PATH} не найден. Сначала обучите модель.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(ACTIONS)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize(IMG_SIZE[::-1]),  # torchvision ждёт (W, H)
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5), (.5,.5,.5)),
    ])

    try:
        cam = Picamera2()
        cam.configure(cam.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}))
        cam.start()
        capture = lambda: cam.capture_array()
    except Exception as e:
        sys.exit(f"Не удалось запустить камеру: {e}")

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
                idx = model(inp).argmax(1).item()
            action = ACTIONS[idx]

            # выполнить действие
            {
                "forward":      forward,
                "backward":     backward,
                "rotate_left":  rotate_left,
                "rotate_right": rotate_right
            }.get(action, stop)()

            print(action)
            if args.preview:
                import matplotlib.pyplot as plt
                plt.imshow(frame); plt.title(action)
                plt.pause(0.001); plt.clf()
    finally:
        cam.stop()
        _all_low()
        GPIO.cleanup()
        print("Exit")

if __name__ == "__main__":
    main()
