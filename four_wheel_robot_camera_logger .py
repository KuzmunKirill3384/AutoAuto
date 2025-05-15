#!/usr/bin/env python3
"""
Four‑wheel robot control on Raspberry Pi.
Adds camera capture & logging with preferred Raspberry Pi library (picamera2),
falling back to OpenCV when Pi‑camera unavailable.
Saved frames + labels → dataset/ for later neural‑net training.
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
    from picamera2 import Picamera2  # libcamera‑based
    _CAM_BACKEND = "picamera2"
except (ImportError, ModuleNotFoundError):
    try:
        import cv2  # type: ignore
        _CAM_BACKEND = "opencv"
    except ImportError:
        cv2 = None  # type: ignore
        _CAM_BACKEND = "none"

# ───────────────────────── GPIO definitions ────────────────────────────
FRONT_LEFT_IN1, FRONT_LEFT_IN2, FRONT_LEFT_EN = 20, 21, 16
FRONT_RIGHT_IN1, FRONT_RIGHT_IN2, FRONT_RIGHT_EN = 13, 26, 19
REAR_RIGHT_IN1, REAR_RIGHT_IN2, REAR_RIGHT_EN = 24, 23, 25
REAR_LEFT_IN1, REAR_LEFT_IN2, REAR_LEFT_EN = 7, 8, 12
PWM_FREQ_HZ = 1000

# ───────────────────────── speed presets ───────────────────────────────
SPEED_PRESETS = {1: 25, 2: 50, 3: 75}

# ───────────────────────── dataset settings ────────────────────────────
DATASET_DIR = Path("dataset"); DATASET_DIR.mkdir(exist_ok=True)
LABELS_CSV = DATASET_DIR / "labels.csv"
FRAME_SAVE_PERIOD = 0.5  # seconds between logs

# ───────────────────────── global state ────────────────────────────────
current_speed_level: int = 1
current_action: Optional[str] = None
_last_frame_ts: float = 0.0

# ───────────────────────── GPIO setup ──────────────────────────────────
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

# ───────────────────────── camera initialisation ───────────────────────
_capture_frame: Callable[[], Optional["numpy.ndarray"]]

if _CAM_BACKEND == "picamera2":
    picam = Picamera2()
    picam.configure(picam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}))
    picam.start()

    def _capture_picam() -> Optional["numpy.ndarray"]:
        try:
            import numpy as np  # local import to avoid dependency issues if not needed
            return picam.capture_array()
        except Exception:
            return None
    _capture_frame = _capture_picam
    print("Camera backend: picamera2 (libcamera)")

elif _CAM_BACKEND == "opencv":
    camera = cv2.VideoCapture(0)  # type: ignore
    if not camera.isOpened():
        print("Warning: camera not detected – logging disabled")
        _CAM_BACKEND = "none"
    else:
        def _capture_cv2() -> Optional["numpy.ndarray"]:
            ret, frame = camera.read()
            return frame if ret else None
        _capture_frame = _capture_cv2  # type: ignore
        print("Camera backend: OpenCV")

if _CAM_BACKEND == "none":
    def _capture_none():
        return None
    _capture_frame = _capture_none  # type: ignore
    print("Camera disabled: no backend available")

# ───────────────────────── pygame / input ──────────────────────────────
pygame.init(); pygame.joystick.init()

# ───────────────────────── helpers ─────────────────────────────────────

def _save_frame(frame, action: str, speed_level: int):
    import cv2 as _cv2
    timestamp = int(time.time() * 1000)
    fname = f"{timestamp}_{action}_{speed_level}.jpg"
    _cv2.imwrite(str(DATASET_DIR / fname), frame)
    with LABELS_CSV.open("a") as fp:
        print(f"{fname},{action},{speed_level},{timestamp}", file=fp)


def _maybe_log_frame():
    global _last_frame_ts
    frame = _capture_frame()
    if frame is None or current_action is None:
        return
    now = time.time()
    if now - _last_frame_ts >= FRAME_SAVE_PERIOD:
        _save_frame(frame, current_action, current_speed_level)
        _last_frame_ts = now


def _apply_pwm_duty(dc: int):
    for pwm in (front_left_pwm, front_right_pwm, rear_left_pwm, rear_right_pwm):
        pwm.ChangeDutyCycle(dc)

# ───────────────────────── motion primitives ───────────────────────────

def set_speed(level: int):
    global current_speed_level
    if level in SPEED_PRESETS:
        current_speed_level = level
        _apply_pwm_duty(SPEED_PRESETS[level])
        print(f"Speed set to {['low','medium','high'][level-1]}")


def _all_low():
    for pin in (FRONT_LEFT_IN1, FRONT_LEFT_IN2, FRONT_RIGHT_IN1, FRONT_RIGHT_IN2,
                REAR_LEFT_IN1, REAR_LEFT_IN2, REAR_RIGHT_IN1, REAR_RIGHT_IN2):
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
    GPIO.output(FRONT_LEFT_IN1, GPIO.LOW);   GPIO.output(FRONT_LEFT_IN2, GPIO.HIGH)
    GPIO.output(FRONT_RIGHT_IN1, GPIO.LOW);  GPIO.output(FRONT_RIGHT_IN2, GPIO.HIGH)
    GPIO.output(REAR_LEFT_IN1, GPIO.LOW);    GPIO.output(REAR_LEFT_IN2, GPIO.HIGH)
    GPIO.output(REAR_RIGHT_IN1, GPIO.LOW);   GPIO.output(REAR_RIGHT_IN2, GPIO.HIGH)
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
    GPIO.output(FRONT_LEFT_IN1, GPIO.HIGH);  GPIO.output(FRONT_LEFT_IN2, GPIO.LOW)
    GPIO.output(FRONT_RIGHT_IN1, GPIO.LOW);  GPIO.output(FRONT_RIGHT_IN2, GPIO.HIGH)
    GPIO.output(REAR_LEFT_IN1, GPIO.HIGH);   GPIO.output(REAR_LEFT_IN2, GPIO.LOW)
    GPIO.output(REAR_RIGHT_IN1, GPIO.LOW);   GPIO.output(REAR_RIGHT_IN2, GPIO.HIGH)
    current_action = "rotate_right"

# ───────────────────────── joystick processing ─────────────────────────

def handle_joystick_movement(x_axis: float, y_axis: float):
    deadzone = 0.2
    if abs(x_axis) < deadzone and abs(y_axis) < deadzone:
        stop(); return
    if y_axis < -deadzone: forward()
    elif y_axis > deadzone: backward()
    elif x_axis < -deadzone: rotate_left()
    elif x_axis > deadzone: rotate_right()

# ───────────────────────── main loop ───────────────────────────────────

def main():
    joystick_count = pygame.joystick.get_count()
    keyboard_mode = joystick_count == 0

    if keyboard_mode:
        print("Keyboard mode – press w/a/s/d/z for movement, x to exit")
        print("(If script runs non‑interactively, keep the window focused and use arrow keys)")
    else:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Detected gamepad: {joystick.get_name()}")

    running = True
    try:
        while running:
            if keyboard_mode:
                # Poll pygame keyboard to avoid blocking input() (which raises EOFError in non‑TTY runs)
                pygame.event.pump()
                kstate = pygame.key.get_pressed()
                if kstate[pygame.K_w]: forward()
                elif kstate[pygame.K_s]: stop()
                elif kstate[pygame.K_a]: rotate_left()
                elif kstate[pygame.K_d]: rotate_right()
                elif kstate[pygame.K_z]: backward()
                elif kstate[pygame.K_1]: set_speed(1)
                elif kstate[pygame.K_2]: set_speed(2)
                elif kstate[pygame.K_3]: set_speed(3)
                elif kstate[pygame.K_x]: running = False
            else:
                for event in pygame.event.get():
                    if event.type == pygame.JOYBUTTONDOWN:
                        if event.button == 0: set_speed(1)
                        elif event.button == 1: set_speed(2)
                        elif event.button == 2: set_speed(3)
                        elif event.button == 7: running = False
                if running:
                    x_axis = joystick.get_axis(0)
                    y_axis = joystick.get_axis(1)
                    handle_joystick_movement(x_axis, y_axis)
            _maybe_log_frame()
            sleep(0.05)
    except KeyboardInterrupt:
        pass
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
