#!/usr/bin/env python3
"""
Four‑wheel robot controller for Raspberry Pi.
Uses gamepad (pygame) for driving and Picamera2/OpenCV for optional dataset logging.
**Keyboard control removed** – script requires a connected joystick/gamepad.
"""
import os
import sys
import time
from pathlib import Path
from typing import Optional, Callable
from time import sleep

import RPi.GPIO as GPIO
import pygame

# ───────────────────────── camera selection ────────────────────────────
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

# ───────────────────────── GPIO pin map ────────────────────────────────
FRONT_LEFT_IN1, FRONT_LEFT_IN2, FRONT_LEFT_EN   = 20, 21, 16
FRONT_RIGHT_IN1, FRONT_RIGHT_IN2, FRONT_RIGHT_EN = 13, 26, 19
REAR_RIGHT_IN1, REAR_RIGHT_IN2, REAR_RIGHT_EN   = 24, 23, 25
REAR_LEFT_IN1,  REAR_LEFT_IN2,  REAR_LEFT_EN    = 7,  8,  12
PWM_FREQ_HZ = 1000

SPEED_PRESETS = {1: 25, 2: 50, 3: 75}

DATASET_DIR = Path("dataset"); DATASET_DIR.mkdir(exist_ok=True)
LABELS_CSV = DATASET_DIR / "labels.csv"
FRAME_SAVE_PERIOD = 0.5

current_speed_level: int = 1
current_action: Optional[str] = None
_last_frame: float = 0.0

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

def _setup_motor(in1: int, in2: int, en: int):
    GPIO.setup(in1, GPIO.OUT); GPIO.setup(in2, GPIO.OUT); GPIO.setup(en, GPIO.OUT)
    GPIO.output(in1, GPIO.LOW); GPIO.output(in2, GPIO.LOW)
    pwm = GPIO.PWM(en, PWM_FREQ_HZ); pwm.start(SPEED_PRESETS[current_speed_level])
    return pwm

front_left_pwm  = _setup_motor(FRONT_LEFT_IN1, FRONT_LEFT_IN2, FRONT_LEFT_EN)
front_right_pwm = _setup_motor(FRONT_RIGHT_IN1, FRONT_RIGHT_IN2, FRONT_RIGHT_EN)
rear_left_pwm   = _setup_motor(REAR_LEFT_IN1,  REAR_LEFT_IN2,  REAR_LEFT_EN)
rear_right_pwm  = _setup_motor(REAR_RIGHT_IN1, REAR_RIGHT_IN2, REAR_RIGHT_EN)

# ───────────────────────── camera init ────────────────────────────────
CaptureFunc = Callable[[], Optional["numpy.ndarray"]]
_capture_frame: CaptureFunc

if _CAM_BACKEND == "picamera2":
    picam = Picamera2()
    picam.configure(picam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}))
    picam.start()
    def _capture_picam():
        try:
            import numpy as np  # noqa
            return picam.capture_array()
        except Exception:
            return None
    _capture_frame = _capture_picam
    print("Camera backend: picamera2")
elif _CAM_BACKEND == "opencv":
    camera = cv2.VideoCapture(0)  # type: ignore
    if camera.isOpened():
        def _capture_cv2():
            ret, frm = camera.read(); return frm if ret else None
        _capture_frame = _capture_cv2  # type: ignore
        print("Camera backend: OpenCV")
    else:
        _CAM_BACKEND = "none"

if _CAM_BACKEND == "none":
    def _capture_none():
        return None
    _capture_frame = _capture_none  # type: ignore
    print("Camera disabled")

# ───────────────────────── helpers ────────────────────────────────────

def _apply_pwm(dc: int):
    for pwm in (front_left_pwm, front_right_pwm, rear_left_pwm, rear_right_pwm):
        pwm.ChangeDutyCycle(dc)

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
    global _last_frame
    if current_action is None:
        return
    frame = _capture_frame()
    now = time.time()
    if frame is not None and now - _last_frame >= FRAME_SAVE_PERIOD:
        _save_frame(frame, current_action)
        _last_frame = now


def _all_low():
    for pin in (FRONT_LEFT_IN1, FRONT_LEFT_IN2, FRONT_RIGHT_IN1, FRONT_RIGHT_IN2,
                REAR_LEFT_IN1,  REAR_LEFT_IN2,  REAR_RIGHT_IN1, REAR_RIGHT_IN2):
        GPIO.output(pin, GPIO.LOW)

def stop():
    global current_action
    _all_low(); current_action = "stop"

def forward():
    global current_action
    GPIO.output(FRONT_LEFT_IN1, GPIO.HIGH);  GPIO.output(FRONT_LEFT_IN2, GPIO.LOW)
    GPIO.output(FRONT_RIGHT_IN1, GPIO.HIGH); GPIO.output(FRONT_RIGHT_IN2, GPIO.LOW)
    GPIO.output(REAR_LEFT_IN1, GPIO.HIGH);   GPIO.output(REAR_LEFT_IN2, GPIO.LOW)
    GPIO.output(REAR_RIGHT_IN1, GPIO.HIGH);  GPIO.output(REAR_RIGHT_IN2, GPIO.LOW)
    current_action = "forward"

def backward():
    global current_action
    GPIO.output(FRONT_LEFT_IN1, GPIO.LOW);  GPIO.output(FRONT_LEFT_IN2, GPIO.HIGH)
    GPIO.output(FRONT_RIGHT_IN1, GPIO.LOW); GPIO.output(FRONT_RIGHT_IN2, GPIO.HIGH)
    GPIO.output(REAR_LEFT_IN1, GPIO.LOW);   GPIO.output(REAR_LEFT_IN2, GPIO.HIGH)
    GPIO.output(REAR_RIGHT_IN1, GPIO.LOW);  GPIO.output(REAR_RIGHT_IN2, GPIO.HIGH)
    current_action = "backward"

def rotate_left():
    global current_action
    GPIO.output(FRONT_LEFT_IN1, GPIO.LOW);   GPIO.output(FRONT_LEFT_IN2, GPIO.HIGH)
    GPIO.output(FRONT_RIGHT_IN1, GPIO.HIGH); GPIO.output(FRONT_RIGHT_IN2, GPIO.LOW)
    GPIO.output(REAR_LEFT_IN1, GPIO.LOW);    GPIO.output(REAR_LEFT_IN2, GPIO.HIGH)
    GPIO.output(REAR_RIGHT_IN1, GPIO.HIGH);  GPIO.output(REAR_RIGHT_IN2, GPIO.LOW)
    current_action = "rotate_left"

def rotate_right():
    global current_action
    GPIO.output(FRONT_LEFT_IN1, GPIO.HIGH); GPIO.output(FRONT_LEFT_IN2, GPIO.LOW)
    GPIO.output(FRONT_RIGHT_IN1, GPIO.LOW); GPIO.output(FRONT_RIGHT_IN2, GPIO.HIGH)
    GPIO.output(REAR_LEFT_IN1, GPIO.HIGH);  GPIO.output(REAR_LEFT_IN2, GPIO.LOW)
    GPIO.output(REAR_RIGHT_IN1, GPIO.LOW);  GPIO.output(REAR_RIGHT_IN2, GPIO.HIGH)
    current_action = "rotate_right"

# ───────────────────────── main loop ──────────────────────────────────

def main():
    if pygame.joystick.get_count() == 0:
        print("No gamepad detected – exiting")
        return
    joystick = pygame.joystick.Joystick(0); joystick.init()
    print("Gamepad:", joystick.get_name())

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 0: set_speed(1)
                    elif event.button == 1: set_speed(2)
                    elif event.button == 2: set_speed(3)
                    elif event.button == 7: running = False
            if not running:
                break
            x_axis = joystick.get_axis(0)
            y_axis = joystick.get_axis(1)
            # Deadzone handling inside
            if abs(x_axis) < 0.2 and abs(y_axis) < 0.2:
                stop()
            elif y_axis < -0.2: forward()
            elif y_axis > 0.2: backward()
            elif x_axis < -0.2: rotate_left()
            elif x_axis > 0.2: rotate_right()
            _maybe_log()
            sleep(0.05)
    finally:
        if _CAM_BACKEND == "picamera2":
            picam.stop()
        elif _CAM_BACKEND == "opencv" and 'camera' in globals():
            camera.release()
        pygame.quit()
        for pwm in (front_left_pwm, front_right_pwm, rear_left_pwm, rear_right_pwm):
            pwm.stop()
        GPIO.cleanup()
        print("Shutdown complete – dataset saved to", DATASET_DIR)


if __name__ == "__main__":
    main()
