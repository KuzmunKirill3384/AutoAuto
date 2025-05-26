#!/usr/bin/env python3
"""
Game-pad controlled four-wheel robot with dataset logger.

Changes vs. original version
────────────────────────────
• «Reverse» включается только при почти полном отклонении стика назад
  ( y_axis > BACKWARD_TRIGGER = 0.8 ).
• Повороты требуют сильного отклонения вбок
  ( |x_axis| > ROTATE_TRIGGER = 0.5 ).
• Всё остальное — как было: логирование кадров, выбор скорости кнопками,
  выключение логирования кнопкой 3, выход кнопкой 7.
"""

import os, sys, time
from pathlib import Path
from typing import Optional, Callable
from time import sleep

import RPi.GPIO as GPIO
import pygame

# ───────── camera back-end autodetect ──────────────────────
try:
    from picamera2 import Picamera2
    _CAM_BACKEND = "picamera2"
except (ImportError, ModuleNotFoundError):
    try:
        import cv2  # type: ignore
        _CAM_BACKEND = "opencv"
    except ImportError:
        cv2 = None  # type: ignore
        _CAM_BACKEND = "none"

# ───────── GPIO pin map ────────────────────────────────────
FL_IN1, FL_IN2, FL_EN = 20, 21, 16
FR_IN1, FR_IN2, FR_EN = 13, 26, 19
RR_IN1, RR_IN2, RR_EN = 24, 23, 25
RL_IN1, RL_IN2, RL_EN = 7,  8,  12
PWM_FREQ_HZ = 1000

# ───────── input thresholds ────────────────────────────────
DEADZONE         = 0.20   # «шумовая» зона
BACKWARD_TRIGGER = 0.80   # reverse только при y > 0.80
ROTATE_TRIGGER   = 0.50   # поворот при |x| > 0.50

# ───────── dataset logging settings ────────────────────────
SPEED_PRESETS = {1: 25, 2: 50, 3: 75}
DATASET_DIR   = Path("dataset"); DATASET_DIR.mkdir(exist_ok=True)
LABELS_CSV    = DATASET_DIR / "labels.csv"

current_speed_level: int = 1
current_action: Optional[str] = None
logging_enabled: bool = False     # toggle кнопкой 3

# ───────── GPIO initialisation ─────────────────────────────
GPIO.setmode(GPIO.BCM); GPIO.setwarnings(False)

def _setup_motor(in1: int, in2: int, en: int):
    GPIO.setup(in1, GPIO.OUT); GPIO.setup(in2, GPIO.OUT); GPIO.setup(en, GPIO.OUT)
    GPIO.output(in1, GPIO.LOW); GPIO.output(in2, GPIO.LOW)
    pwm = GPIO.PWM(en, PWM_FREQ_HZ)
    pwm.start(SPEED_PRESETS[current_speed_level])
    return pwm

fl_pwm = _setup_motor(FL_IN1, FL_IN2, FL_EN)
fr_pwm = _setup_motor(FR_IN1, FR_IN2, FR_EN)
rl_pwm = _setup_motor(RL_IN1, RL_IN2, RL_EN)
rr_pwm = _setup_motor(RR_IN1, RR_IN2, RR_EN)

# ───────── camera init ─────────────────────────────────────
CaptureFunc = Callable[[], Optional["numpy.ndarray"]]
def _capture_none(): return None  # fallback
_capture_frame: CaptureFunc = _capture_none  # default

if _CAM_BACKEND == "picamera2":
    picam = Picamera2()
    picam.configure(picam.create_video_configuration(main={"size": (640,480),
                                                           "format":"RGB888"}))
    picam.start()
    _capture_frame = lambda: picam.capture_array()
    print("Camera backend: picamera2")

elif _CAM_BACKEND == "opencv":
    camera = cv2.VideoCapture(0)  # type: ignore
    if camera.isOpened():
        _capture_frame = lambda: camera.read()[1]  # type: ignore
        print("Camera backend: OpenCV")
    else:
        _CAM_BACKEND = "none"

if _CAM_BACKEND == "none":
    print("Camera disabled")

# ───────── helpers ─────────────────────────────────────────
def _apply_pwm(dc: int):
    for p in (fl_pwm, fr_pwm, rl_pwm, rr_pwm):
        p.ChangeDutyCycle(dc)

def set_speed(level: int):
    global current_speed_level
    if level in SPEED_PRESETS:
        current_speed_level = level
        _apply_pwm(SPEED_PRESETS[level])
        print("Speed →", level)

def _save_frame(frame, action: str):
    import cv2 as _cv2
    ts = int(time.time()*1000)
    fname = f"{ts}_{action}_{current_speed_level}.jpg"
    _cv2.imwrite(str(DATASET_DIR / fname), frame)
    with LABELS_CSV.open("a") as fp:
        print(f"{fname},{action},{current_speed_level},{ts}", file=fp)

def _maybe_log():
    if logging_enabled and current_action:
        frame = _capture_frame()
        if frame is not None:
            _save_frame(frame, current_action)

def _all_low():
    for pin in (FL_IN1, FL_IN2, FR_IN1, FR_IN2,
                RL_IN1, RL_IN2, RR_IN1, RR_IN2):
        GPIO.output(pin, GPIO.LOW)

# ───────── motion primitives ───────────────────────────────
def stop():
    global current_action
    _all_low(); current_action = "stop"

def forward():
    global current_action
    _all_low()
    GPIO.output(FL_IN1,GPIO.HIGH); GPIO.output(RL_IN1,GPIO.HIGH)
    GPIO.output(FR_IN1,GPIO.HIGH); GPIO.output(RR_IN1,GPIO.HIGH)
    current_action = "forward"

def backward():
    global current_action
    _all_low()
    GPIO.output(FL_IN2,GPIO.HIGH); GPIO.output(RL_IN2,GPIO.HIGH)
    GPIO.output(FR_IN2,GPIO.HIGH); GPIO.output(RR_IN2,GPIO.HIGH)
    current_action = "backward"

def rotate_left():
    global current_action
    _all_low()
    GPIO.output(FR_IN1,GPIO.HIGH); GPIO.output(RR_IN1,GPIO.HIGH)
    GPIO.output(FL_IN2,GPIO.HIGH); GPIO.output(RL_IN2,GPIO.HIGH)
    current_action = "rotate_left"

def rotate_right():
    global current_action
    _all_low()
    GPIO.output(FL_IN1,GPIO.HIGH); GPIO.output(RL_IN1,GPIO.HIGH)
    GPIO.output(FR_IN2,GPIO.HIGH); GPIO.output(RR_IN2,GPIO.HIGH)
    current_action = "rotate_right"

def toggle_logging():
    global logging_enabled
    logging_enabled = not logging_enabled
    print("Logging", "ON" if logging_enabled else "OFF")

# ───────── main loop ───────────────────────────────────────
def main():
    pygame.init(); pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        sys.exit("No gamepad detected")
    js = pygame.joystick.Joystick(0); js.init()
    pygame.event.clear()
    print("Gamepad:", js.get_name())

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    if   event.button == 0: set_speed(1)
                    elif event.button == 1: set_speed(2)
                    elif event.button == 2: set_speed(3)
                    elif event.button == 3: toggle_logging()
                    elif event.button == 7: running = False
            if not running: break

            x = js.get_axis(0)
            y = js.get_axis(1)

            if abs(x) < DEADZONE and abs(y) < DEADZONE:
                stop()
            elif y < -DEADZONE:              # стик вперёд
                forward()
            elif y > BACKWARD_TRIGGER:       # стик почти внизу
                backward()
            elif x < -ROTATE_TRIGGER:        # сильный влево
                rotate_left()
            elif x >  ROTATE_TRIGGER:        # сильный вправо
                rotate_right()
            else:
                stop()

            _maybe_log()
            sleep(0.005)
    finally:
        if _CAM_BACKEND == "picamera2":
            picam.stop()
        elif _CAM_BACKEND == "opencv" and 'camera' in globals():
            camera.release()
        pygame.quit()
        for p in (fl_pwm, fr_pwm, rl_pwm, rr_pwm): p.stop()
        GPIO.cleanup()
        print("Shutdown complete – dataset at", DATASET_DIR)

if __name__ == "__main__":
    main()
