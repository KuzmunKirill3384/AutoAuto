#!/usr/bin/env python3
"""
Autonomous 4-wheel robot (Picamera2).

• Загружает обученный TorchScript-файл MobileNetV3-Small (robot_action_cnn_script.pt).
• Каждые 0.01 с берёт кадр 640×480, ресайзит до 128×96, нормализует,
  прогнозирует одно из четырёх действий и даёт команду моторам по GPIO.
"""

from __future__ import annotations
import sys, time, argparse
from pathlib import Path
from time import sleep
from typing import Optional

import RPi.GPIO as GPIO
import torch
from torchvision import transforms
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
SPEED = 45                       # duty-cycle %

# ─── модель и параметры инференса ───────────────────────────
SCRIPT_PATH  = Path("robot_action_cnn.pt")
IMG_SIZE     = (96, 128)         # H×W
ACTIONS      = ["forward", "backward", "rotate_left", "rotate_right"]
FRAME_PERIOD = 0.01              # минимальная пауза между кадрами

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

def stop():
    _all_low()

def forward():
    _all_low()
    GPIO.output(FRONT_LEFT_IN1,  GPIO.HIGH)
    GPIO.output(REAR_LEFT_IN1,   GPIO.HIGH)
    GPIO.output(FRONT_RIGHT_IN1, GPIO.HIGH)
    GPIO.output(REAR_RIGHT_IN1,  GPIO.HIGH)

def backward():
    _all_low()
    GPIO.output(FRONT_LEFT_IN2,  GPIO.HIGH)
    GPIO.output(REAR_LEFT_IN2,   GPIO.HIGH)
    GPIO.output(FRONT_RIGHT_IN2, GPIO.HIGH)
    GPIO.output(REAR_RIGHT_IN2,  GPIO.HIGH)

def rotate_left():
    _all_low()
    GPIO.output(FRONT_RIGHT_IN1, GPIO.HIGH)
    GPIO.output(REAR_RIGHT_IN1,  GPIO.HIGH)
    GPIO.output(FRONT_LEFT_IN2,  GPIO.HIGH)
    GPIO.output(REAR_LEFT_IN2,   GPIO.HIGH)

def rotate_right():
    _all_low()
    GPIO.output(FRONT_LEFT_IN1,  GPIO.HIGH)
    GPIO.output(REAR_LEFT_IN1,   GPIO.HIGH)
    GPIO.output(FRONT_RIGHT_IN2, GPIO.HIGH)
    GPIO.output(REAR_RIGHT_IN2,  GPIO.HIGH)

# ─── основной код ───────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="Показ кадра и предсказания (нужен X11)")
    args = parser.parse_args()

    if not SCRIPT_PATH.exists():
        sys.exit(f"{SCRIPT_PATH} не найден. Сначала обучите модель.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = torch.jit.load(str(SCRIPT_PATH), map_location=device)
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize(IMG_SIZE[::-1]),  # (W,H)
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5), (.5,.5,.5)),
    ])

    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    ))
    cam.start()
    capture = lambda: cam.capture_array()

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
                action = ACTIONS[int(model(inp).argmax(1))]

            if action == "forward":
                forward()
            elif action == "backward":
                backward()
            elif action == "rotate_left":
                rotate_left()
            elif action == "rotate_right":
                rotate_right()
            else:
                stop()

            print(action)
            if args.preview:
                import matplotlib.pyplot as plt
                plt.imshow(frame)
                plt.title(action)
                plt.pause(0.001)
                plt.clf()
    finally:
        cam.stop()
        _all_low()
        GPIO.cleanup()
        print("Exit")

if __name__ == "__main__":
    main()
