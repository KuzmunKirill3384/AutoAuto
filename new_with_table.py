import RPi.GPIO as GPIO
import pygame
from time import sleep

# ─────────────── Пины моторов ────────────────

front_left_in1,  front_left_in2,  front_left_en   = 20, 21, 16
front_right_in1, front_right_in2, front_right_en  = 13, 26, 19
rear_left_in1,   rear_left_in2,   rear_left_en    = 24, 23, 25
rear_right_in1,  rear_right_in2,  rear_right_en   = 7,  8,  12

# ─────────────── Инициализация GPIO ────────────────

GPIO.setmode(GPIO.BCM)

pins = (
    front_left_in1, front_left_in2, front_left_en,
    front_right_in1, front_right_in2, front_right_en,
    rear_left_in1, rear_left_in2, rear_left_en,
    rear_right_in1, rear_right_in2, rear_right_en
)

for pin in pins:
    GPIO.setup(pin, GPIO.OUT)

GPIO.output(
    [
        front_left_in1, front_left_in2,
        front_right_in1, front_right_in2,
        rear_left_in1, rear_left_in2,
        rear_right_in1, rear_right_in2
    ],
    GPIO.LOW
)

fl_pwm = GPIO.PWM(front_left_en,  1000)
fr_pwm = GPIO.PWM(front_right_en, 1000)
rl_pwm = GPIO.PWM(rear_left_en,   1000)
rr_pwm = GPIO.PWM(rear_right_en,  1000)

fl_pwm.start(25)
fr_pwm.start(25)
rl_pwm.start(25)
rr_pwm.start(25)

# ─────────────── Таблица направлений ────────────────

DIRECTIONS = {
    'front_left':  {'forward': (GPIO.HIGH, GPIO.LOW), 'backward': (GPIO.LOW, GPIO.HIGH)},
    'front_right': {'forward': (GPIO.LOW, GPIO.HIGH), 'backward': (GPIO.HIGH, GPIO.LOW)},
    'rear_left':   {'forward': (GPIO.HIGH, GPIO.LOW), 'backward': (GPIO.LOW, GPIO.HIGH)},
    'rear_right':  {'forward': (GPIO.LOW, GPIO.HIGH), 'backward': (GPIO.HIGH, GPIO.LOW)},
}

WHEELS = {
    'front_left':  (front_left_in1,  front_left_in2),
    'front_right': (front_right_in1, front_right_in2),
    'rear_left':   (rear_left_in1,   rear_left_in2),
    'rear_right':  (rear_right_in1,  rear_right_in2),
}

def set_wheel(wheel, direction):
    in1, in2 = WHEELS[wheel]
    level1, level2 = DIRECTIONS[wheel][direction]
    GPIO.output(in1, level1)
    GPIO.output(in2, level2)

def forward():
    for wheel in WHEELS:
        set_wheel(wheel, 'forward')
    print("Forward")

def backward():
    for wheel in WHEELS:
        set_wheel(wheel, 'backward')
    print("Backward")

def rotate_left():
    for wheel in ['front_left', 'rear_left']:
        set_wheel(wheel, 'backward')
    for wheel in ['front_right', 'rear_right']:
        set_wheel(wheel, 'forward')
    print("Rotate left")

def rotate_right():
    for wheel in ['front_left', 'rear_left']:
        set_wheel(wheel, 'forward')
    for wheel in ['front_right', 'rear_right']:
        set_wheel(wheel, 'backward')
    print("Rotate right")

def stop():
    for in1, in2 in WHEELS.values():
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
    print("Stop")

# ─────────────── Управление и джойстик ────────────────

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise RuntimeError("Gamepad not found")

js = pygame.joystick.Joystick(0)
js.init()

current_speed = 25
def set_speed_level(level):
    global current_speed
    duty = {1: 25, 2: 50, 3: 75}.get(level, current_speed)
    current_speed = duty
    for pwm in (fl_pwm, fr_pwm, rl_pwm, rr_pwm):
        pwm.ChangeDutyCycle(duty)
    print(f"Speed set to {level}")

set_speed_level(1)

deadzone = 0.2
running = True

print("Use stick ↑/↓ for Fwd/Bwd, ←/→ for Rotate, buttons 1–3 for speed, Start to exit.")

while running:
    for ev in pygame.event.get():
        if ev.type == pygame.JOYBUTTONDOWN:
            if ev.button == 0:
                set_speed_level(1)
            if ev.button == 1:
                set_speed_level(2)
            if ev.button == 2:
                set_speed_level(3)
            if ev.button == 7:
                running = False

    x = js.get_axis(0)
    y = js.get_axis(1)

    if x < -deadzone:
        rotate_left()
    elif x > deadzone:
        rotate_right()
    elif y < -deadzone:
        forward()
    elif y > deadzone:
        backward()
    else:
        stop()

    sleep(0.05)

# ─────────────── Очистка ────────────────

fl_pwm.stop()
fr_pwm.stop()
rl_pwm.stop()
rr_pwm.stop()

GPIO.cleanup()
pygame.quit()

print("Done.")
