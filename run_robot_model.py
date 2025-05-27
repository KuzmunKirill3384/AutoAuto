#!/usr/bin/env python3
"""
Autonomous 4-wheel robot (Picamera2) — инференс под модель без backward.
"""

import sys
import time
from pathlib import Path

import RPi.GPIO as GPIO
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw

# Пробуем импортировать Picamera2, если нет - сообщаем об ошибке и выходим
try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("sudo apt install python3-picamera2")

# Настройка GPIO пинов для управления моторами
FL_IN1, FL_IN2, FL_EN = 20, 21, 16
FR_IN1, FR_IN2, FR_EN = 13, 26, 19
RL_IN1, RL_IN2, RL_EN = 7,  8,  12
RR_IN1, RR_IN2, RR_EN = 24, 23, 25
PWM_FREQ_HZ = 1000      # Частота ШИМ
BASE_SPEED  = 30        # Скорость для прямого движения, %
FAST_SPEED  = 45        # Скорость для вращения, %

# Инициализируем GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Список для хранения объектов PWM
_pwms = []

def _setup_motor(in1, in2, en):
    # Настраиваем пины и стартуем ШИМ
    GPIO.setup(in1, GPIO.OUT)
    GPIO.setup(in2, GPIO.OUT)
    GPIO.setup(en, GPIO.OUT)
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)
    pwm = GPIO.PWM(en, PWM_FREQ_HZ)
    pwm.start(BASE_SPEED)
    _pwms.append(pwm)

# Настраиваем все четыре мотора
for pins in (
    (FL_IN1, FL_IN2, FL_EN),
    (FR_IN1, FR_IN2, FR_EN),
    (RL_IN1, RL_IN2, RL_EN),
    (RR_IN1, RR_IN2, RR_EN),
):
    _setup_motor(*pins)

# Функция для установки общего ШИМ-значения для всех моторов
def _set_speed(dc: int):
    for pwm in _pwms:
        pwm.ChangeDutyCycle(dc)

# Останавливаем все моторы (LOW на всех входах)
def _all_low():
    for pin in (FL_IN1, FL_IN2, FR_IN1, FR_IN2,
                RL_IN1, RL_IN2, RR_IN1, RR_IN2):
        GPIO.output(pin, GPIO.LOW)

# Примитивы движения:
def stop():
    _all_low()  # полностью стоп

def forward():
    _set_speed(BASE_SPEED)
    _all_low()
    # Включаем прямое движение на всех моторах
    GPIO.output(FL_IN1, GPIO.HIGH)
    GPIO.output(FR_IN1, GPIO.HIGH)
    GPIO.output(RL_IN1, GPIO.HIGH)
    GPIO.output(RR_IN1, GPIO.HIGH)

def rotate_left():
    _set_speed(FAST_SPEED)
    _all_low()
    # Вращение влево: два правых вперед, два левых назад
    GPIO.output(FR_IN1, GPIO.HIGH)
    GPIO.output(RR_IN1, GPIO.HIGH)
    GPIO.output(FL_IN2, GPIO.HIGH)
    GPIO.output(RL_IN2, GPIO.HIGH)

def rotate_right():
    _set_speed(FAST_SPEED)
    _all_low()
    # Вращение вправо: два левых вперед, два правых назад
    GPIO.output(FL_IN1, GPIO.HIGH)
    GPIO.output(RL_IN1, GPIO.HIGH)
    GPIO.output(FR_IN2, GPIO.HIGH)
    GPIO.output(RR_IN2, GPIO.HIGH)

# Путь к файлу модели и параметры обработки изображения
MODEL_PATH = Path("robot_action_cnn.pth")
IMG_SIZE   = (96, 128)            # Высота × Ширина
ACTIONS    = ["forward", "rotate_left", "rotate_right"]
FRAME_DT   = 0.02                # Пауза между кадрами

# Маска для затемнения нижней части изображения
class BottomMask:
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        y0 = int(h * 0.60)  # оставляем верхние 60%
        out = img.copy()
        ImageDraw.Draw(out).rectangle([(0, y0), (w, h)], fill=0)
        return out

# Собираем пайплайн препроцессинга как при обучении
tfms = transforms.Compose([
    BottomMask(),                         # затемняем низ кадра
    transforms.Resize((IMG_SIZE[0], IMG_SIZE[1])),  # изменяем размер
    transforms.ToTensor(),                # в тензор
    transforms.Normalize((.5, .5, .5), (.5, .5, .5)),  # нормировка
])

# Функция сборки сети (тот же TinyCNN, но MobileNetV3)
def build_model(num_cls: int):
    net = models.mobilenet_v3_small(weights=None)
    # Меняем последний линейный слой под количество классов
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_cls)
    return net

# Главная функция
def main():
    # Проверяем наличие файла модели
    if not MODEL_PATH.exists():
        sys.exit(f"{MODEL_PATH} not found — please place your .pth here")

    # Делаем инференс на CPU (нет CUDA на Pi)
    device = torch.device("cpu")
    model  = build_model(len(ACTIONS)).to(device)
    state  = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Настраиваем камеру
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    ))
    cam.start()
    capture = lambda: cam.capture_array()

    next_time = time.time()
    try:
        while True:
            # Получаем кадр из камеры
            frame = capture()
            img   = Image.fromarray(frame)
            inp   = tfms(img).unsqueeze(0).to(device)

            # Предсказание
            with torch.no_grad():
                idx = model(inp).argmax(1).item()
            action = ACTIONS[idx] if 0 <= idx < len(ACTIONS) else None

            # Выполняем движение
            if   action == "forward":      forward()
            elif action == "rotate_left":  rotate_left()
            elif action == "rotate_right": rotate_right()
            else:                            stop()

            print("→", action or "stop")

            # Ждём до следующего кадра
            next_time += FRAME_DT
            time.sleep(max(0, next_time - time.time()))
    finally:
        cam.stop()
        stop()
        for p in _pwms:
            p.stop()
        GPIO.cleanup()
        print("Exit")

if __name__ == "__main__":
    main()
