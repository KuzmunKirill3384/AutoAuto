#!/usr/bin/env python3
"""
Autonomous 4-wheel robot (Picamera2) — только вперёд и повороты.

Использует TorchScript-модель (3 класса):
    • forward
    • rotate_left
    • rotate_right
"""

from __future__ import annotations
import sys, time, argparse
from pathlib import Path
from typing import Callable

import RPi.GPIO as GPIO
import torch
from torchvision import transforms
from PIL import Image

# ─── Picamera2 ─────────────────────────────────────────────
try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("Picamera2 не установлена: sudo apt install python3-picamera2")

# ─── GPIO pin-map (из «рабочего» скрипта) ─────────────────
FL_IN1, FL_IN2, FL_EN = 20, 21, 16
FR_IN1, FR_IN2, FR_EN = 13, 26, 19
RR_IN1, RR_IN2, RR_EN = 24, 23, 25
RL_IN1, RL_IN2, RL_EN = 7,  8,  12
PWM_FREQ_HZ, SPEED = 1000, 30      # %

# ─── модель и инференс ────────────────────────────────────
SCRIPT_PATH  = Path("robot_action_cnn_script.pt")   # TorchScript (3 класса)
IMG_SIZE     = (96, 128)                            # H×W
ACTIONS      = ["forward", "rotate_left", "rotate_right"]
FRAME_PERIOD = 0.01                                 # c

# ─── GPIO init ────────────────────────────────────────────
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
def _setup_motor(in1: int, in2: int, en: int):
    GPIO.setup(in1, GPIO.OUT); GPIO.setup(in2, GPIO.OUT); GPIO.setup(en, GPIO.OUT)
    GPIO.output(in1, GPIO.LOW); GPIO.output(in2, GPIO.LOW)
    pwm = GPIO.PWM(en, PWM_FREQ_HZ); pwm.start(SPEED)
    return pwm
for pins in ((FL_IN1, FL_IN2, FL_EN), (FR_IN1, FR_IN2, FR_EN),
             (RL_IN1, RL_IN2, RL_EN), (RR_IN1, RR_IN2, RR_EN)):
    _setup_motor(*pins)

# ─── движения (идентично «рабочему» скрипту) ──────────────
def _all_low():
    for pin in (FL_IN1, FL_IN2, FR_IN1, FR_IN2,
                RL_IN1, RL_IN2, RR_IN1, RR_IN2):
        GPIO.output(pin, GPIO.LOW)

def forward():
    _all_low()
    GPIO.output(FL_IN1, GPIO.HIGH); GPIO.output(RL_IN1, GPIO.HIGH)
    GPIO.output(FR_IN1, GPIO.HIGH); GPIO.output(RR_IN1, GPIO.HIGH)

def rotate_left():
    _all_low()
    GPIO.output(FR_IN1, GPIO.HIGH); GPIO.output(RR_IN1, GPIO.HIGH)
    GPIO.output(FL_IN2, GPIO.HIGH); GPIO.output(RL_IN2, GPIO.HIGH)

def rotate_right():
    _all_low()
    GPIO.output(FL_IN1, GPIO.HIGH); GPIO.output(RL_IN1, GPIO.HIGH)
    GPIO.output(FR_IN2, GPIO.HIGH); GPIO.output(RR_IN2, GPIO.HIGH)

def stop():
    _all_low()

# ─── MAIN ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="Показ кадра и предсказания (нужен X-server)")
    args = parser.parse_args()

    if not SCRIPT_PATH.exists():
        sys.exit(f"{SCRIPT_PATH} не найден — сначала сохраните TorchScript.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── загружаем TorchScript ──────────────────────────────
    try:
        model: torch.jit.ScriptModule = torch.jit.load(str(SCRIPT_PATH),
                                                       map_location=device)
    except Exception as e:
        sys.exit(f"Не удалось открыть TorchScript: {e}")
    model.eval()

    # ── проверяем число выходов (прогон dummy-тензора) ────
    with torch.no_grad():
        out_shape = model(torch.zeros(1, 3, *IMG_SIZE, device=device)).shape
    if out_shape[1] != 3:
        sys.exit(f"⚠ TorchScript имеет {out_shape[1]} классов, а нужно 3 "
                 f"(forward / rotate_left / rotate_right). "
                 f"Пересохраните модель.")

    tfms = transforms.Compose([
        transforms.Resize(IMG_SIZE[::-1]),  # torchvision ждёт (W,H)
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    ])

    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}))
    cam.start(); capture: Callable[[], "np.ndarray"] = lambda: cam.capture_array()

    last = 0.0
    try:
        while True:
            if time.time() - last < FRAME_PERIOD:
                continue
            last = time.time()

            frame = capture()
            inp = tfms(Image.fromarray(frame)).unsqueeze(0).to(device)

            with torch.no_grad():
                idx = int(model(inp).argmax(1))

            # safety-guard (Idx должно быть 0..2)
            action = ACTIONS[idx] if idx < len(ACTIONS) else "stop"

            {
                "forward":      forward,
                "rotate_left":  rotate_left,
                "rotate_right": rotate_right,
                "stop":         stop
            }[action]()

            print(action)
            if args.preview:
                import matplotlib.pyplot as plt
                plt.imshow(frame); plt.title(action); plt.pause(0.001); plt.clf()
    finally:
        cam.stop()
        stop()
        GPIO.cleanup()
        print("Exit")

if __name__ == "__main__":
    main()
