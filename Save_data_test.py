#!/usr/bin/env python3
import time
import cv2
import RPi.GPIO as GPIO
import numpy as np
import os
from datetime import datetime

# Обновленная конфигурация GPIO для L298N и сервоприводов согласно указанным пинам
# Servo 1
IN1 = 11  # Пин для направления 1 первого сервопривода
IN2 = 12  # Пин для направления 2 первого сервопривода
ENA = 33  # Пин для ШИМ (напряжение) первого сервопривода

# Servo 2
IN3 = 15  # Пин для направления 1 второго сервопривода
IN4 = 16  # Пин для направления 2 второго сервопривода
ENB = 35  # Пин для ШИМ (напряжение) второго сервопривода

# Аналоговые входы для чтения напряжения (через АЦП)
try:
    import Adafruit_ADS1x15

    adc = Adafruit_ADS1x15.ADS1115()
    ADC_AVAILABLE = True
except ImportError:
    print("ADS1115 library not found. Voltage measurements will be simulated.")
    ADC_AVAILABLE = False

# Создание структуры директорий для хранения данных
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Создание или открытие файла labels.csv
LABELS_FILE = os.path.join(DATASET_DIR, "labels.csv")
if not os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, 'w') as f:
        f.write("image_name,servo1_direction,servo1_voltage,servo2_direction,servo2_voltage\n")

# Настройка GPIO
GPIO.setmode(GPIO.BOARD)  # Используем нумерацию пинов платы, а не BCM
GPIO.setwarnings(False)

# Настройка пинов для L298N
GPIO.setup(IN1, GPIO.IN)
GPIO.setup(IN2, GPIO.IN)
GPIO.setup(IN3, GPIO.IN)
GPIO.setup(IN4, GPIO.IN)
GPIO.setup(ENA, GPIO.IN)
GPIO.setup(ENB, GPIO.IN)


# Настройка камеры
def setup_camera():
    cap = cv2.VideoCapture(0)  # Использование камеры по умолчанию
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        exit()

    # Настройка разрешения камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


# Функция для определения направления движения сервопривода
def get_servo_direction(input1, input2):
    in1_state = GPIO.input(input1)
    in2_state = GPIO.input(input2)

    if in1_state == GPIO.HIGH and in2_state == GPIO.LOW:
        return "forward"
    elif in1_state == GPIO.LOW and in2_state == GPIO.HIGH:
        return "backward"
    else:
        return "stop"


# Функция для измерения напряжения на сервоприводах
def get_servo_voltage(ena_pin, adc_channel=0):
    if ADC_AVAILABLE:
        # Чтение значения с АЦП
        value = adc.read_adc(adc_channel, gain=1)
        # Преобразование значения АЦП в напряжение
        voltage = value * (5.0 / 32767)  # Для ADS1115 с диапазоном ±5В
        return voltage
    else:
        # Имитация чтения напряжения для тестирования
        duty_cycle = GPIO.input(ena_pin)
        # Имитируем напряжение на основе ШИМ
        return duty_cycle * 5.0 / 100


# Функция для захвата и сохранения кадра с камеры
def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось захватить кадр")
        return None
    return frame


# Функция для сохранения данных в новой структуре
def save_data(frame, servo1_direction, servo1_voltage, servo2_direction, servo2_voltage, index):
    # Форматирование имени файла с ведущими нулями
    frame_name = f"frame_{index:06d}.jpg"
    frame_path = os.path.join(IMAGES_DIR, frame_name)

    # Сохранение кадра
    cv2.imwrite(frame_path, frame)

    # Добавление записи в labels.csv
    with open(LABELS_FILE, 'a') as f:
        f.write(f"{frame_name},{servo1_direction},{servo1_voltage:.2f},{servo2_direction},{servo2_voltage:.2f}\n")

    return frame_name


# Основная функция
def main():
    print("Запуск программы сбора данных для обучения нейросети...")

    # Настраиваем камеру
    cap = setup_camera()

    # Определяем начальный индекс, чтобы не перезаписать существующие файлы
    existing_files = os.listdir(IMAGES_DIR)
    if existing_files:
        # Получаем максимальный индекс из существующих файлов
        max_index = max([int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith('frame_')])
        index = max_index + 1
    else:
        index = 1

    try:
        while True:
            # Получаем данные о сервоприводах
            servo1_direction = get_servo_direction(IN1, IN2)
            servo1_voltage = get_servo_voltage(ENA, adc_channel=0)

            servo2_direction = get_servo_direction(IN3, IN4)
            servo2_voltage = get_servo_voltage(ENB, adc_channel=1)

            # Захватываем кадр с камеры
            frame = capture_frame(cap)
            if frame is not None:
                # Сохраняем данные
                frame_name = save_data(frame, servo1_direction, servo1_voltage, servo2_direction, servo2_voltage, index)

                # Отображаем информацию в консоли
                print(
                    f"Сохранено: {frame_name} | Servo1: {servo1_direction} ({servo1_voltage:.2f}V), Servo2: {servo2_direction} ({servo2_voltage:.2f}V)")

                # Отображаем кадр с информацией для визуального контроля
                info_frame = frame.copy()
                cv2.putText(info_frame, f"Frame: {index:06d}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(info_frame, f"S1: {servo1_direction} {servo1_voltage:.2f}V", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(info_frame, f"S2: {servo2_direction} {servo2_voltage:.2f}V", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Data Collection", info_frame)

                # Нажмите 'q' для выхода
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                index += 1

            # Задержка для контроля частоты сбора данных
            time.sleep(0.1)  # 10 кадров в секунду

    except KeyboardInterrupt:
        print("\nПрограмма остановлена пользователем")

    finally:
        # Освобождаем ресурсы
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print(f"Собрано {index - 1} образцов данных.")
        print(f"Изображения сохранены в: {IMAGES_DIR}")
        print(f"Метки данных сохранены в: {LABELS_FILE}")


if __name__ == "__main__":
    main()