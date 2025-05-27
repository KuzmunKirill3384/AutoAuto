#!/usr/bin/env python3
"""
Autonomous 4-wheel robot (Picamera2) — инференс под «новую» модель.

• MobileNetV3-Small, 4 класса.
• Препроцессинг 1-в-1 как в обучении (маска 40 % низа, resize 96×128).
• forward → 30 % PWM, остальные движения → 45 %.
"""

from __future__ import annotations
import sys, time
from pathlib import Path

import RPi.GPIO as GPIO
import torch, torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw
try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("sudo apt install python3-picamera2")

# ─── GPIO ------------------------------------------------------------------
FL_IN1, FL_IN2, FL_EN = 20, 21, 16
FR_IN1, FR_IN2, FR_EN = 13, 26, 19
RR_IN1, RR_IN2, RR_EN = 24, 23, 25
RL_IN1, RL_IN2, RL_EN = 7,  8,  12
PWM_FREQ_HZ   = 1000
BASE_SPEED    = 30   # %
FAST_SPEED    = 45   # %

GPIO.setmode(GPIO.BCM); GPIO.setwarnings(False)
_pwms: list[GPIO.PWM] = []
def _setup_motor(in1, in2, en):
    GPIO.setup(in1, GPIO.OUT); GPIO.setup(in2, GPIO.OUT); GPIO.setup(en, GPIO.OUT)
    GPIO.output(in1, GPIO.LOW); GPIO.output(in2, GPIO.LOW)
    pwm = GPIO.PWM(en, PWM_FREQ_HZ); pwm.start(BASE_SPEED)
    _pwms.append(pwm)
for p in ((FL_IN1, FL_IN2, FL_EN), (FR_IN1, FR_IN2, FR_EN),
          (RL_IN1, RL_IN2, RL_EN), (RR_IN1, RR_IN2, RR_EN)):
    _setup_motor(*p)

def _set_speed(dc:int):
    for pwm in _pwms: pwm.ChangeDutyCycle(dc)

# ─── движ-примитивы ---------------------------------------------------------
def _all_low():
    for pin in (FL_IN1, FL_IN2, FR_IN1, FR_IN2,
                RL_IN1, RL_IN2, RR_IN1, RR_IN2):
        GPIO.output(pin, GPIO.LOW)

def stop():           _all_low()
def forward():        _set_speed(BASE_SPEED); _all_low(); GPIO.output(FL_IN1,1); GPIO.output(RL_IN1,1); GPIO.output(FR_IN1,1); GPIO.output(RR_IN1,1)
def backward():       _set_speed(FAST_SPEED); _all_low(); GPIO.output(FL_IN2,1); GPIO.output(RL_IN2,1); GPIO.output(FR_IN2,1); GPIO.output(RR_IN2,1)
def rotate_left():    _set_speed(FAST_SPEED); _all_low(); GPIO.output(FR_IN1,1); GPIO.output(RR_IN1,1); GPIO.output(FL_IN2,1); GPIO.output(RL_IN2,1)
def rotate_right():   _set_speed(FAST_SPEED); _all_low(); GPIO.output(FL_IN1,1); GPIO.output(RL_IN1,1); GPIO.output(FR_IN2,1); GPIO.output(RR_IN2,1)

# ─── модель / трансформы ----------------------------------------------------
MODEL_PATH  = Path("robot_action_cnn.pth")
IMG_SIZE    = (96, 128)
ACTIONS     = ["forward", "backward", "rotate_left", "rotate_right"]
FRAME_DT    = 0.02      # ~50 fps (можно увеличить/уменьшить)

class BottomMask:
    """Закрашивает 40 % высоты снизу чёрным."""
    def __call__(self, img: Image.Image):
        w, h = img.size
        y0 = int(h*0.60)             # оставляем верхние 60 %
        img = img.copy()
        ImageDraw.Draw(img).rectangle([(0, y0), (w, h)], fill=0)
        return img

tfms = transforms.Compose([
    BottomMask(),                             # ← важно!
    transforms.Resize(IMG_SIZE[::-1]),
    transforms.ToTensor(),
    transforms.Normalize((.5,.5,.5), (.5,.5,.5)),
])

def build_model(num_cls=4):
    net = models.mobilenet_v3_small(weights=None)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_cls)
    return net

# ─── MAIN -------------------------------------------------------------------
def main():
    if not MODEL_PATH.exists():
        sys.exit(f"{MODEL_PATH} not found")

    dev = torch.device("cpu")    # на Pi — CPU; CUDA недоступна
    model = build_model(len(ACTIONS)).to(dev)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
    model.eval()

    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}))
    cam.start()
    capture = lambda: cam.capture_array()

    next_t = time.time()
    try:
        while True:
            frame = capture()
            img_t = tfms(Image.fromarray(frame)).unsqueeze(0).to(dev)

            with torch.no_grad():
                act = ACTIONS[int(model(img_t).argmax(1))]

            {"forward": forward,
             "backward": backward,
             "rotate_left": rotate_left,
             "rotate_right": rotate_right}.get(act, stop)()

            print(act)
            next_t += FRAME_DT
            time.sleep(max(0, next_t - time.time()))
    finally:
        cam.stop(); stop()
        for p in _pwms: p.stop()
        GPIO.cleanup()
        print("Exit")

if __name__ == "__main__":
    main()
