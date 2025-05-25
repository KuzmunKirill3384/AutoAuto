#!/usr/bin/env python3
"""
Real-time inference script for the 4-wheel robot.

• Loads the trained model weights (robot_action_cnn.pth).
• Captures live frames from the camera (Picamera2 ▸ OpenCV ▸ None).
• Predicts one of four actions: forward / backward / rotate_left / rotate_right.
• Prints prediction every 0.2 s (same cadence as dataset logging).
• Optionally lights an LED on GPIO18 while the script is running.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import RPi.GPIO as GPIO  # type: ignore
except (ImportError, RuntimeError):
    GPIO = None

# ────────────── ML imports ──────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
except ImportError:
    sys.stderr.write("❌  PyTorch not found. Install with: pip3 install torch torchvision\n"); raise

try:
    from PIL import Image
except ImportError:
    sys.stderr.write("❌  Pillow missing. Install: pip3 install pillow\n"); raise

# ────────────── Camera backend ──────────
try:
    from picamera2 import Picamera2  # type: ignore
    _CAM_BACKEND = "picamera2"
except (ImportError, ModuleNotFoundError):
    try:
        import cv2  # type: ignore
        _CAM_BACKEND = "opencv"
    except ImportError:
        _CAM_BACKEND = "none"
        cv2 = None  # type: ignore

# ────────────── config ──────────────────
MODEL_PATH = Path("robot_action_cnn.pth")
IMG_SIZE = (96, 128)  # H×W — must match training
LED_PIN = 18          # None → disable
ACTIONS = [
    "forward",
    "backward",
    "rotate_left",
    "rotate_right",
]
FRAME_PERIOD = 0.2    # seconds between predictions

# ────────────── Net definition (same as training) ───────────
class TinyCNN(nn.Module):
    def __init__(self, n_classes=len(ACTIONS)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        dummy = torch.zeros(1, 3, *IMG_SIZE)
        with torch.no_grad():
            n_flat = self._conv(dummy).numel()
        self.fc1 = nn.Linear(n_flat, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def _conv(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._conv(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ────────────── helpers ──────────────────

def _prepare_led():
    if GPIO is None or LED_PIN is None:
        return None
    GPIO.setmode(GPIO.BCM); GPIO.setwarnings(False)
    GPIO.setup(LED_PIN, GPIO.OUT); GPIO.output(LED_PIN, GPIO.HIGH)
    return LED_PIN

def _release_led(pin: Optional[int]):
    if pin is not None and GPIO is not None:
        GPIO.output(pin, GPIO.LOW)
        GPIO.cleanup(pin)

# ────────────── main ────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time robot action classifier")
    parser.add_argument("--headless", action="store_true", help="Don't show OpenCV preview window")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        sys.exit(f"❌  Model not found: {MODEL_PATH}. Train it first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize(IMG_SIZE[::-1]),
        transforms.ToTensor(),
    ])

    # Camera init
    if _CAM_BACKEND == "picamera2":
        picam = Picamera2()
        picam.configure(picam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}))
        picam.start()
        capture = lambda: picam.capture_array()
    elif _CAM_BACKEND == "opencv":
        cap = cv2.VideoCapture(0)  # type: ignore
        if not cap.isOpened():
            sys.exit("❌  Unable to open /dev/video0")
        capture = lambda: cap.read()[1]
    else:
        sys.exit("❌  No camera backend available")

    led_pin = _prepare_led()
    try:
        last = 0.0
        while True:
            frame = capture()
            if frame is None:
                print("⚠️  Empty frame, skipping"); continue

            now = time.time()
            if now - last < FRAME_PERIOD:
                continue
            last = now

            # Convert frame to PIL Image (OpenCV gives BGR)
            if _CAM_BACKEND == "opencv":
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore
            else:
                frame_rgb = frame
            img = Image.fromarray(frame_rgb)
            input_tensor = tfms(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                pred_idx = logits.argmax(1).item()
                action = ACTIONS[pred_idx]

            print(f"→ {action}")

            # Show preview
            if not args.headless and _CAM_BACKEND == "opencv":
                cv2.putText(frame, action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # type: ignore
                cv2.imshow("Robot view", frame)  # type: ignore
                if cv2.waitKey(1) & 0xFF == ord('q'):  # type: ignore
                    break
    finally:
        if _CAM_BACKEND == "picamera2":
            picam.stop()
        elif _CAM_BACKEND == "opencv":
            cap.release()  # type: ignore
            if not args.headless:
                cv2.destroyAllWindows()  # type: ignore
        _release_led(led_pin)
        print("👋  Bye!")

if __name__ == "__main__":
    main()
