#!/usr/bin/env python3
"""
Autonomous 4-wheel robot runner using a trained TinyCNN.

• Загружает модель robot_action_cnn.pth и сеть TinyCNN.
• Снимает кадры с камеры (Picamera2 ▶ OpenCV ▶ None).
• Каждые 0.2 с предсказывает действие: forward/backward/rotate_left/rotate_right.
• Вместо геймпада вызывает функции управления:
    forward(), backward(), rotate_left(), rotate_right(), stop().
• Поддерживает светодиод на GPIO18 как индикатор работы.

Запуск:
    python3 auto_drive.py [--headless]

"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from time import sleep
from typing import Optional, Callable

import RPi.GPIO as GPIO  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# камера
try:
    from picamera2 import Picamera2  # type: ignore
    _CAM_BACKEND = "picamera2"
except ImportError:
    try:
        import cv2  # type: ignore
        _CAM_BACKEND = "opencv"
    except ImportError:
        _CAM_BACKEND = "none"
        cv2 = None  # type: ignore

# GPIO пины управления моторами
FRONT_LEFT_IN1, FRONT_LEFT_IN2, FRONT_LEFT_EN   = 20, 21, 16
FRONT_RIGHT_IN1, FRONT_RIGHT_IN2, FRONT_RIGHT_EN = 13, 26, 19
REAR_RIGHT_IN1, REAR_RIGHT_IN2, REAR_RIGHT_EN   = 24, 23, 25
REAR_LEFT_IN1,  REAR_LEFT_IN2,  REAR_LEFT_EN    = 7,  8,  12
PWM_FREQ_HZ = 1000

# прелоад скорости (0–100)
SPEED = 50

# путь к модели
MODEL_PATH = Path("robot_action_cnn.pth")
IMG_SIZE = (96, 128)
LED_PIN = 18  # None → без LED
ACTIONS = ["forward","backward","rotate_left","rotate_right"]
FRAME_PERIOD = 0.2

# ────────────── инициализация GPIO моторов ──────────────
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

def _setup_motor(in1, in2, en):
    GPIO.setup(in1, GPIO.OUT); GPIO.setup(in2, GPIO.OUT); GPIO.setup(en, GPIO.OUT)
    GPIO.output(in1, GPIO.LOW); GPIO.output(in2, GPIO.LOW)
    pwm = GPIO.PWM(en, PWM_FREQ_HZ); pwm.start(SPEED)
    return pwm

front_left_pwm  = _setup_motor(FRONT_LEFT_IN1, FRONT_LEFT_IN2, FRONT_LEFT_EN)
front_right_pwm = _setup_motor(FRONT_RIGHT_IN1, FRONT_RIGHT_IN2, FRONT_RIGHT_EN)
rear_left_pwm   = _setup_motor(REAR_LEFT_IN1,  REAR_LEFT_IN2,  REAR_LEFT_EN)
rear_right_pwm  = _setup_motor(REAR_RIGHT_IN1, REAR_RIGHT_IN2, REAR_RIGHT_EN)

# ────────────── ф-ии управления ──────────────

def _all_low():
    for pin in (FRONT_LEFT_IN1, FRONT_LEFT_IN2,
                FRONT_RIGHT_IN1, FRONT_RIGHT_IN2,
                REAR_LEFT_IN1, REAR_LEFT_IN2,
                REAR_RIGHT_IN1, REAR_RIGHT_IN2):
        GPIO.output(pin, GPIO.LOW)

current_action: Optional[str] = None

def stop():
    global current_action
    _all_low(); current_action = None

def forward():
    global current_action
    _all_low()
    GPIO.output(FRONT_LEFT_IN1, GPIO.HIGH);  GPIO.output(FRONT_RIGHT_IN1, GPIO.HIGH)
    GPIO.output(REAR_LEFT_IN1, GPIO.HIGH);   GPIO.output(REAR_RIGHT_IN1, GPIO.HIGH)
    current_action = "forward"

def backward():
    global current_action
    _all_low()
    GPIO.output(FRONT_LEFT_IN2, GPIO.HIGH);  GPIO.output(FRONT_RIGHT_IN2, GPIO.HIGH)
    GPIO.output(REAR_LEFT_IN2, GPIO.HIGH);   GPIO.output(REAR_RIGHT_IN2, GPIO.HIGH)
    current_action = "backward"

def rotate_left():
    global current_action
    _all_low()
    GPIO.output(FRONT_RIGHT_IN1, GPIO.HIGH)
    GPIO.output(REAR_RIGHT_IN1, GPIO.HIGH)
    GPIO.output(FRONT_LEFT_IN2, GPIO.HIGH)
    GPIO.output(REAR_LEFT_IN2, GPIO.HIGH)
    current_action = "rotate_left"

def rotate_right():
    global current_action
    _all_low()
    GPIO.output(FRONT_LEFT_IN1, GPIO.HIGH)
    GPIO.output(REAR_LEFT_IN1, GPIO.HIGH)
    GPIO.output(FRONT_RIGHT_IN2, GPIO.HIGH)
    GPIO.output(REAR_RIGHT_IN2, GPIO.HIGH)
    current_action = "rotate_right"

# ────────────── LED-индикатор ──────────────

def _prepare_led():
    if GPIO is None or LED_PIN is None:
        return None
    GPIO.setup(LED_PIN, GPIO.OUT); GPIO.output(LED_PIN, GPIO.HIGH)
    return LED_PIN

def _release_led(pin: Optional[int]):
    if pin is not None:
        GPIO.output(pin, GPIO.LOW); GPIO.cleanup(pin)

# ────────────── сеть и колбэки ──────────────
class TinyCNN(nn.Module):
    def __init__(self, n_classes=len(ACTIONS)):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        dummy = torch.zeros(1,3,*IMG_SIZE)
        with torch.no_grad(): n_flat = self._conv(dummy).numel()
        self.fc1 = nn.Linear(n_flat,128); self.fc2 = nn.Linear(128,n_classes)
    def _conv(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.pool(F.relu(self.conv3(x)))
    def forward(self,x):
        x = self._conv(x).flatten(1)
        return self.fc2(F.relu(self.fc1(x)))

# ────────────── main ──────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        sys.exit("Model not found, train first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    tfms = transforms.Compose([transforms.Resize(IMG_SIZE[::-1]), transforms.ToTensor()])

    # init camera
    if _CAM_BACKEND=="picamera2":
        cam = Picamera2(); cam.configure(cam.create_video_configuration(main={"size":(640,480),"format":"RGB888"})); cam.start(); capture=lambda:cam.capture_array()
    elif _CAM_BACKEND=="opencv":
        cap=cv2.VideoCapture(0)
        if not cap.isOpened(): sys.exit("Cannot open camera")
        capture=lambda:cap.read()[1]
    else:
        sys.exit("No camera backend")

    led_pin = _prepare_led()
    last=0.0
    try:
        while True:
            now=time.time()
            if now-last<FRAME_PERIOD: continue
            last=now
            frame=capture()
            if frame is None: continue
            # convert
            if _CAM_BACKEND=="opencv": frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img=Image.fromarray(frame); inp=tfms(img).unsqueeze(0).to(device)
            with torch.no_grad(): idx=model(inp).argmax(1).item()
            act=ACTIONS[idx]
            # call control
            if act=="forward": forward()
            elif act=="backward": backward()
            elif act=="rotate_left": rotate_left()
            elif act=="rotate_right": rotate_right()
            else: stop()
            # preview
            if not args.headless and _CAM_BACKEND=="opencv":
                cv2.putText(frame,act,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                cv2.imshow("view",frame)
                if cv2.waitKey(1)&0xFF==ord('q'): break
            sleep(0.05)
    finally:
        if _CAM_BACKEND=="picamera2": cam.stop()
        elif _CAM_BACKEND=="opencv": cap.release(); cv2.destroyAllWindows()
        _release_led(led_pin)
        _all_low(); GPIO.cleanup()
        print("Exit")

if __name__=="__main__": main()
