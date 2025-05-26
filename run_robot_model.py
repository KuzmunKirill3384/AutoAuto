#!/usr/bin/env python3
"""
Autonomous 4-wheel robot (Picamera2).

• MobileNetV3-Small, 4 выхода.
• forward едет на 30 % PWM, backward и повороты — на 45 %.
"""

from __future__ import annotations
import sys, time, argparse
from pathlib import Path

import RPi.GPIO as GPIO
import torch, torch.nn as nn
from torchvision import transforms, models
from PIL import Image
try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("sudo apt install python3-picamera2")

# ─── GPIO pins ─────────────────────────────────────────────
FL_IN1, FL_IN2, FL_EN = 20, 21, 16
FR_IN1, FR_IN2, FR_EN = 13, 26, 19
RR_IN1, RR_IN2, RR_EN = 24, 23, 25
RL_IN1, RL_IN2, RL_EN = 7,  8,  12
PWM_FREQ_HZ   = 1000
BASE_SPEED    = 30   # %
FAST_SPEED    = 45   # %

# ─── модель / инференс ─────────────────────────────────────
MODEL_PATH   = Path("robot_action_cnn.pth")
IMG_SIZE     = (96, 128)
ACTIONS      = ["forward", "backward", "rotate_left", "rotate_right"]
FRAME_PERIOD = 0.01

# ─── GPIO init ─────────────────────────────────────────────
GPIO.setmode(GPIO.BCM); GPIO.setwarnings(False)
_pwms: list[GPIO.PWM] = []
def _setup_motor(in1, in2, en):
    GPIO.setup(in1, GPIO.OUT); GPIO.setup(in2, GPIO.OUT); GPIO.setup(en, GPIO.OUT)
    GPIO.output(in1, GPIO.LOW); GPIO.output(in2, GPIO.LOW)
    pwm = GPIO.PWM(en, PWM_FREQ_HZ); pwm.start(BASE_SPEED)
    _pwms.append(pwm)
for pins in ((FL_IN1, FL_IN2, FL_EN), (FR_IN1, FR_IN2, FR_EN),
             (RL_IN1, RL_IN2, RL_EN), (RR_IN1, RR_IN2, RR_EN)):
    _setup_motor(*pins)

def _set_speed(dc: int):
    for pwm in _pwms:
        pwm.ChangeDutyCycle(dc)

# ─── low-level moves ───────────────────────────────────────
def _all_low():
    for pin in (FL_IN1, FL_IN2, FR_IN1, FR_IN2,
                RL_IN1, RL_IN2, RR_IN1, RR_IN2):
        GPIO.output(pin, GPIO.LOW)

def stop(): _all_low()

def forward():
    _set_speed(BASE_SPEED)
    _all_low()
    GPIO.output(FL_IN1,GPIO.HIGH); GPIO.output(RL_IN1,GPIO.HIGH)
    GPIO.output(FR_IN1,GPIO.HIGH); GPIO.output(RR_IN1,GPIO.HIGH)

def backward():
    _set_speed(FAST_SPEED)
    _all_low()
    GPIO.output(FL_IN2,GPIO.HIGH); GPIO.output(RL_IN2,GPIO.HIGH)
    GPIO.output(FR_IN2,GPIO.HIGH); GPIO.output(RR_IN2,GPIO.HIGH)

def rotate_left():
    _set_speed(FAST_SPEED)
    _all_low()
    GPIO.output(FR_IN1,GPIO.HIGH); GPIO.output(RR_IN1,GPIO.HIGH)
    GPIO.output(FL_IN2,GPIO.HIGH); GPIO.output(RL_IN2,GPIO.HIGH)

def rotate_right():
    _set_speed(FAST_SPEED)
    _all_low()
    GPIO.output(FL_IN1,GPIO.HIGH); GPIO.output(RL_IN1,GPIO.HIGH)
    GPIO.output(FR_IN2,GPIO.HIGH); GPIO.output(RR_IN2,GPIO.HIGH)

# ─── build model ───────────────────────────────────────────
def build_model(num_cls=4):
    net = models.mobilenet_v3_small(weights=None, width_mult=1.0)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_cls)
    return net

# ─── MAIN ──────────────────────────────────────────────────
def main():
    if not MODEL_PATH.exists():
        sys.exit(f"{MODEL_PATH} не найден – обучите модель.")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(ACTIONS)).to(dev)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize(IMG_SIZE[::-1]),
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5), (.5,.5,.5)),
    ])

    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}))
    cam.start(); capture = lambda: cam.capture_array()

    last = 0.0
    try:
        while True:
            if time.time() - last < FRAME_PERIOD:
                continue
            last = time.time()

            frame = capture()
            inp   = tfms(Image.fromarray(frame)).unsqueeze(0).to(dev)
            with torch.no_grad():
                act = ACTIONS[int(model(inp).argmax(1))]

            {"forward": forward,
             "backward": backward,
             "rotate_left": rotate_left,
             "rotate_right": rotate_right}.get(act, stop)()

            print(act)
    finally:
        cam.stop()
        stop()
        for pwm in _pwms: pwm.stop()
        GPIO.cleanup()
        print("Exit")

if __name__ == "__main__":
    main()
