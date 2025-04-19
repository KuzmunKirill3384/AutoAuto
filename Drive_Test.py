#!/usr/bin/env python3

import RPi.GPIO as GPIO
import time

# Настройка пинов для первого мотора
motor1_pin1 = 11  # IN1 на драйвере L298N
motor1_pin2 = 12  # IN2 на драйвере L298N
enable1_pin = 33  # Пин ENA для первого мотора

# Настройка пинов для второго мотора
motor2_pin1 = 15  # IN3 на драйвере L298N
motor2_pin2 = 16  # IN4 на драйвере L298N
enable2_pin = 35  # Пин ENB для второго мотора

# Настройка GPIO
GPIO.setmode(GPIO.BOARD)  # Использование нумерации пинов платы (BOARD)
GPIO.setwarnings(False)

# Настройка пинов как выходы
GPIO.setup(motor1_pin1, GPIO.OUT)
GPIO.setup(motor1_pin2, GPIO.OUT)
GPIO.setup(enable1_pin, GPIO.OUT)

GPIO.setup(motor2_pin1, GPIO.OUT)
GPIO.setup(motor2_pin2, GPIO.OUT)
GPIO.setup(enable2_pin, GPIO.OUT)

# Создание PWM объектов для управления скоростью моторов
motor1_pwm = GPIO.PWM(enable1_pin, 100)  # Частота 100 Гц
motor2_pwm = GPIO.PWM(enable2_pin, 100)  # Частота 100 Гц

# Запуск PWM со скважностью 0 (остановлено)
motor1_pwm.start(0)
motor2_pwm.start(0)


def set_motor_speed(motor_number, speed):
    """
    Установка скорости и направления мотора.
    motor_number: 1 или 2 (номер мотора)
    speed: от -100 до 100 (отрицательные значения для обратного вращения)
    """
    if motor_number == 1:
        motor_pwm = motor1_pwm
        pin1 = motor1_pin1
        pin2 = motor1_pin2
    else:
        motor_pwm = motor2_pwm
        pin1 = motor2_pin1
        pin2 = motor2_pin2

    # Ограничение скорости в диапазоне -100 до 100
    speed = max(-100, min(100, speed))

    # Установка направления вращения
    if speed > 0:
        GPIO.output(pin1, GPIO.HIGH)
        GPIO.output(pin2, GPIO.LOW)
        motor_pwm.ChangeDutyCycle(speed)
    elif speed < 0:
        GPIO.output(pin1, GPIO.LOW)
        GPIO.output(pin2, GPIO.HIGH)
        motor_pwm.ChangeDutyCycle(abs(speed))
    else:
        GPIO.output(pin1, GPIO.LOW)
        GPIO.output(pin2, GPIO.LOW)
        motor_pwm.ChangeDutyCycle(0)


def stop_motors():
    """Остановка обоих моторов"""
    set_motor_speed(1, 0)
    set_motor_speed(2, 0)


def cleanup():
    """Очистка GPIO"""
    motor1_pwm.stop()
    motor2_pwm.stop()
    GPIO.cleanup()


# Пример использования
if __name__ == "__main__":
    try:
        print("Тестирование моторов")

        # Мотор 1 вперед на 70% скорости
        print("Мотор 1 вперед")
        set_motor_speed(1, 70)
        time.sleep(2)

        # Мотор 1 назад на 50% скорости
        print("Мотор 1 назад")
        set_motor_speed(1, -50)
        time.sleep(2)

        # Мотор 2 вперед на 70% скорости
        print("Мотор 2 вперед")
        set_motor_speed(2, 70)
        time.sleep(2)

        # Мотор 2 назад на 50% скорости
        print("Мотор 2 назад")
        set_motor_speed(2, -50)
        time.sleep(2)

        # Оба мотора вперед
        print("Оба мотора вперед")
        set_motor_speed(1, 60)
        set_motor_speed(2, 60)
        time.sleep(3)

        print("Остановка моторов")
        stop_motors()

    except KeyboardInterrupt:
        print("Программа прервана пользователем")
    finally:
        cleanup()
        print("GPIO очищены")