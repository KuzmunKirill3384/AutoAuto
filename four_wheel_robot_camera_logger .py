#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import cv2  # OpenCV for camera
except ImportError:
    cv2 = None  # camera functionality will be disabled if OpenCV missing

import RPi.GPIO as GPIO
import pygame
from time import sleep

# ===== GPIO definitions =====
FRONT_LEFT_IN1, FRONT_LEFT_IN2, FRONT_LEFT_EN = 20, 21, 16
FRONT_RIGHT_IN1, FRONT_RIGHT_IN2, FRONT_RIGHT_EN = 13, 26, 19
REAR_RIGHT_IN1, REAR_RIGHT_IN2, REAR_RIGHT_EN = 24, 23, 25
REAR_LEFT_IN1, REAR_LEFT_IN2, REAR_LEFT_EN = 7, 8, 12
PWM_FREQ_HZ = 1000

# ===== Speed presets =====
SPEED_PRESETS = {
    1: 25,  # low
    2: 50,  # medium
    3: 75,  # high
}

# ===== Dataset settings =====
DATASET_DIR = Path("dataset")
DATASET_DIR.mkdir(exist_ok=True)
LABELS_CSV = DATASET_DIR / "labels.csv"
FRAME_SAVE_PERIOD = 0.5  # seconds between frame saves

# ===== Global mutable state =====
current_speed_level: int = 1
current_action: Optional[str] = None
_last_frame_ts: float = 0.0

# ===== GPIO setup =====
GPIO.setmode(GPIO.BCM)

def _setup_motor(in1: int, in2: int, en: int):
    GPIO.setup(in1, GPIO.OUT)
    GPIO.setup(in2, GPIO.OUT)
    GPIO.setup(en, GPIO.OUT)
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)
    pwm = GPIO.PWM(en, PWM_FREQ_HZ)
    pwm.start(SPEED_PRESETS[current_speed_level])
    return pwm

front_left_pwm = _setup_motor(FRONT_LEFT_IN1, FRONT_LEFT_IN2, FRONT_LEFT_EN)
front_right_pwm = _setup_motor(FRONT_RIGHT_IN1, FRONT_RIGHT_IN2, FRONT_RIGHT_EN)
rear_left_pwm = _setup_motor(REAR_LEFT_IN1, REAR_LEFT_IN2, REAR_LEFT_EN)
rear_right_pwm = _setup_motor(REAR_RIGHT_IN1, REAR_RIGHT_IN2, REAR_RIGHT_EN)

# ===== Camera initialisation =====
if cv2 is not None:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Warning: camera not detected – logging disabled")
        camera = None
else:
    camera = None

# ===== Pygame / input initialisation =====
pygame.init()
pygame.joystick.init()

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _save_frame(frame, action: str, speed_level: int):
    """Save a single video frame with label to disk + csv."""
    timestamp = int(time.time() * 1000)
    fname = f"{timestamp}_{action}_{speed_level}.jpg"
    fpath = DATASET_DIR / fname
    cv2.imwrite(str(fpath), frame)
    with LABELS_CSV.open("a") as fp:
        print(f"{fname},{action},{speed_level},{timestamp}", file=fp)


def _maybe_log_frame():
    global _last_frame_ts
    if camera is None or current_action is None:
        return
    now = time.time()
    if now - _last_frame_ts < FRAME_SAVE_PERIOD:
        return
    ret, frame = camera.read()
    if ret:
        _save_frame(frame, current_action, current_speed_level)
        _last_frame_ts = now


def _apply_pwm_duty(duty: int):
    for pwm in (front_left_pwm, front_right_pwm, rear_left_pwm, rear_right_pwm):
        pwm.ChangeDutyCycle(duty)


# ---------------------------------------------------------------------------
# Motor control helpers (update current_action)
# ---------------------------------------------------------------------------

def set_speed(level: int):
    global current_speed_level
    if level not in SPEED_PRESETS:
        return
    current_speed_level = level
    _apply_pwm_duty(SPEED_PRESETS[level])
    print(f"Speed set to {['low','medium','high'][level-1]}")


def _all_low():
    for pin in (FRONT_LEFT_IN1, FRONT_LEFT_IN2, FRONT_RIGHT_IN1, FRONT_RIGHT_IN2,
                REAR_LEFT_IN1, REAR_LEFT_IN2, REAR_RIGHT_IN1, REAR_RIGHT_IN2):
        GPIO.output(pin, GPIO.LOW)


def stop():
    global current_action
    _all_low()
    current_action = "stop"


def forward():
    global current_action
    GPIO.output(FRONT_LEFT_IN1, GPIO.HIGH); GPIO.output(FRONT_LEFT_IN2, GPIO.LOW)
    GPIO.output(FRONT_RIGHT_IN1, GPIO.HIGH); GPIO.output(FRONT_RIGHT_IN2, GPIO.LOW)
    GPIO.output(REAR_LEFT_IN1, GPIO.HIGH); GPIO.output(REAR_LEFT_IN2, GPIO.LOW)
    GPIO.output(REAR_RIGHT_IN1, GPIO.HIGH); GPIO.output(REAR_RIGHT_IN2, GPIO.LOW)
    current_action = "forward"


def backward():
    global current_action
    GPIO.output(FRONT_LEFT_IN1, GPIO.LOW); GPIO.output(FRONT_LEFT_IN2, GPIO.HIGH)
    GPIO.output(FRONT_RIGHT_IN1, GPIO.LOW); GPIO.output(FRONT_RIGHT_IN2, GPIO.HIGH)
    GPIO.output(REAR_LEFT_IN1, GPIO.LOW); GPIO.output(REAR_LEFT_IN2, GPIO.HIGH)
    GPIO.output(REAR_RIGHT_IN1, GPIO.LOW); GPIO.output(REAR_RIGHT_IN2, GPIO.HIGH)
    current_action = "backward"


def rotate_left():
    global current_action
    GPIO.output(FRONT_LEFT_IN1, GPIO.LOW); GPIO.output(FRONT_LEFT_IN2, GPIO.HIGH)
    GPIO.output(FRONT_RIGHT_IN1, GPIO.HIGH); GPIO.output(FRONT_RIGHT_IN2, GPIO.LOW)
    GPIO.output(REAR_LEFT_IN1, GPIO.LOW); GPIO.output(REAR_LEFT_IN2, GPIO.HIGH)
    GPIO.output(REAR_RIGHT_IN1, GPIO.HIGH); GPIO.output(REAR_RIGHT_IN2, GPIO.LOW)
    current_action = "rotate_left"


def rotate_right():
    global current_action
    GPIO.output(FRONT_LEFT_IN1, GPIO.HIGH); GPIO.output(FRONT_LEFT_IN2, GPIO.LOW)
    GPIO.output(FRONT_RIGHT_IN1, GPIO.LOW); GPIO.output(FRONT_RIGHT_IN2, GPIO.HIGH)
    GPIO.output(REAR_LEFT_IN1, GPIO.HIGH); GPIO.output(REAR_LEFT_IN2, GPIO.LOW)
    GPIO.output(REAR_RIGHT_IN1, GPIO.LOW); GPIO.output(REAR_RIGHT_IN2, GPIO.HIGH)
    current_action = "rotate_right"


def handle_joystick_movement(x_axis: float, y_axis: float):
    deadzone = 0.2
    if abs(x_axis) < deadzone and abs(y_axis) < deadzone:
        stop(); return
    if y_axis < -deadzone:  # forward zone
        forward()
    elif y_axis > deadzone:  # backward zone
        backward()
    elif x_axis < -deadzone:
        rotate_left()
    elif x_axis > deadzone:
        rotate_right()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    joystick_count = pygame.joystick.get_count()
    keyboard_mode = joystick_count == 0
    if keyboard_mode:
        print("Keyboard mode engaged – camera logging active")
    else:
        joystick = pygame.joystick.Joystick(0); joystick.init()
        print(f"Detected gamepad: {joystick.get_name()}")

    running = True
    while running:
        try:
            if keyboard_mode:
                cmd = input().lower()
                if cmd == 'w': forward()
                elif cmd == 's': stop()
                elif cmd == 'a': rotate_left()
                elif cmd == 'd': rotate_right()
                elif cmd == 'z': backward()
                elif cmd in {'1','2','3'}: set_speed(int(cmd))
                elif cmd == 'x': running = False
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
            running = False

    # cleanup
    if camera is not None:
        camera.release()
    pygame.quit()
    for pwm in (front_left_pwm, front_right_pwm, rear_left_pwm, rear_right_pwm):
        pwm.stop()
    GPIO.cleanup()
    print("Shutdown complete – dataset saved to", DATASET_DIR)


if __name__ == "__main__":
    main()
