#!/usr/bin/env python3
"""
Autonomous 4-wheel robot (Picamera2 only).

Каждые 0.2 с берёт кадр, прогоняет через TinyCNN и
вызывает forward / backward / rotate_left / rotate_right / stop.

Мощность моторов увеличена: PWM duty-cycle = 95 %.
"""

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
SPEED = 95           # duty-cycle ≈ 95 % — почти максимум

# ─── модель и параметры инференса ────────────────────────────────────
MODEL_PATH  = Path("robot_action_cnn.pth")
IMG_SIZE    = (96, 128)                 # H×W
ACTIONS     = ["forward","backward","rotate_left","rotate_right"]
FRAME_PERIOD = 0.2                      # с

# ─── GPIO init ───────────────────────────────────────────────────────
GPIO.setmode(GPIO.BCM); GPIO.setwarnings(False)
def _setup_motor(in1, in2, en):
    GPIO.setup(in1, GPIO.OUT); GPIO.setup(in2, GPIO.OUT); GPIO.setup(en, GPIO.OUT)
    GPIO.output(in1, GPIO.LOW); GPIO.output(in2, GPIO.LOW)
    pwm = GPIO.PWM(en, PWM_FREQ_HZ); pwm.start(SPEED)
    return pwm
front_left_pwm  = _setup_motor(FRONT_LEFT_IN1,  FRONT_LEFT_IN2,  FRONT_LEFT_EN)
front_right_pwm = _setup_motor(FRONT_RIGHT_IN1, FRONT_RIGHT_IN2, FRONT_RIGHT_EN)
rear_left_pwm   = _setup_motor(REAR_LEFT_IN1,   REAR_LEFT_IN2,   REAR_LEFT_EN)
rear_right_pwm  = _setup_motor(REAR_RIGHT_IN1,  REAR_RIGHT_IN2,  REAR_RIGHT_EN)

# ─── функции движения ───────────────────────────────────────────────
def _all_low():
    for pin in (FRONT_LEFT_IN1, FRONT_LEFT_IN2, FRONT_RIGHT_IN1, FRONT_RIGHT_IN2,
                REAR_LEFT_IN1, REAR_LEFT_IN2, REAR_RIGHT_IN1, REAR_RIGHT_IN2):
        GPIO.output(pin, GPIO.LOW)

def stop():          _all_low()
def forward():       _all_low(); GPIO.output(FRONT_LEFT_IN1,GPIO.HIGH);  GPIO.output(REAR_LEFT_IN1,GPIO.HIGH);  GPIO.output(FRONT_RIGHT_IN1,GPIO.HIGH); GPIO.output(REAR_RIGHT_IN1,GPIO.HIGH)
def backward():      _all_low(); GPIO.output(FRONT_LEFT_IN2,GPIO.HIGH);  GPIO.output(REAR_LEFT_IN2,GPIO.HIGH);  GPIO.output(FRONT_RIGHT_IN2,GPIO.HIGH); GPIO.output(REAR_RIGHT_IN2,GPIO.HIGH)
def rotate_left():   _all_low(); GPIO.output(FRONT_RIGHT_IN1,GPIO.HIGH); GPIO.output(REAR_RIGHT_IN1,GPIO.HIGH); GPIO.output(FRONT_LEFT_IN2,GPIO.HIGH);  GPIO.output(REAR_LEFT_IN2,GPIO.HIGH)
def rotate_right():  _all_low(); GPIO.output(FRONT_LEFT_IN1,GPIO.HIGH);  GPIO.output(REAR_LEFT_IN1,GPIO.HIGH);  GPIO.output(FRONT_RIGHT_IN2,GPIO.HIGH);GPIO.output(REAR_RIGHT_IN2,GPIO.HIGH)

# ─── сеть TinyCNN ────────────────────────────────────────────────────
class TinyCNN(nn.Module):
    def __init__(self, n_classes=len(ACTIONS)):
        super().__init__()
        self.conv1, self.conv2, self.conv3 = nn.Conv2d(3,16,3,1,1), nn.Conv2d(16,32,3,1,1), nn.Conv2d(32,64,3,1,1)
        self.pool = nn.MaxPool2d(2,2)
        dummy = torch.zeros(1,3,*IMG_SIZE)
        with torch.no_grad(): n_flat = self._conv(dummy).numel()
        self.fc1, self.fc2 = nn.Linear(n_flat,128), nn.Linear(128,n_classes)
    def _conv(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.pool(F.relu(self.conv3(x)))
    def forward(self,x):
        x = self._conv(x).flatten(1)
        return self.fc2(F.relu(self.fc1(x)))

# ─── основной код ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="Окно превью (нужен X11)")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        sys.exit("robot_action_cnn.pth не найден. Сначала обучите модель.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    tfms = transforms.Compose([transforms.Resize(IMG_SIZE[::-1]), transforms.ToTensor()])

    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"size": (640,480), "format":"RGB888"}))
    cam.start(); capture = lambda: cam.capture_array()

    last = 0.0
    try:
        while True:
            if (now := time.time()) - last < FRAME_PERIOD:
                continue
            last = now
            frame = capture()
            img = Image.fromarray(frame)
            inp = tfms(img).unsqueeze(0).to(device)
            with torch.no_grad():
                act_idx = model(inp).argmax(1).item()
            act = ACTIONS[act_idx]

            if   act == "forward":       forward()
            elif act == "backward":      backward()
            elif act == "rotate_left":   rotate_left()
            elif act == "rotate_right":  rotate_right()
            else:                        stop()

            print(act)
            if args.preview:
                import matplotlib.pyplot as plt
                plt.imshow(frame); plt.title(act); plt.pause(0.001); plt.clf()
            sleep(0.05)
    finally:
        cam.stop()
        _all_low(); GPIO.cleanup()
        print("Exit")

if __name__ == "__main__":
    main()
