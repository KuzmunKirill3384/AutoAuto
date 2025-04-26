import RPi.GPIO as GPIO
from time import sleep

# Front wheels GPIO pins (from both_weels_2.py)
front_left_in1 = 20
front_left_in2 = 21
front_left_en = 16

front_right_in1 = 13
front_right_in2 = 26
front_right_en = 19

# Rear wheels GPIO pins (from both_weels.py)
rear_left_in1 = 24
rear_left_in2 = 23
rear_left_en = 25

rear_right_in1 = 7
rear_right_in2 = 8
rear_right_en = 12

# Initialize GPIO
GPIO.setmode(GPIO.BCM)

# Setup front left wheel
GPIO.setup(front_left_in1, GPIO.OUT)
GPIO.setup(front_left_in2, GPIO.OUT)
GPIO.setup(front_left_en, GPIO.OUT)
GPIO.output(front_left_in1, GPIO.LOW)
GPIO.output(front_left_in2, GPIO.LOW)
front_left_pwm = GPIO.PWM(front_left_en, 1000)
front_left_pwm.start(25)

# Setup front right wheel
GPIO.setup(front_right_in1, GPIO.OUT)
GPIO.setup(front_right_in2, GPIO.OUT)
GPIO.setup(front_right_en, GPIO.OUT)
GPIO.output(front_right_in1, GPIO.LOW)
GPIO.output(front_right_in2, GPIO.LOW)
front_right_pwm = GPIO.PWM(front_right_en, 1000)
front_right_pwm.start(25)

# Setup rear left wheel
GPIO.setup(rear_left_in1, GPIO.OUT)
GPIO.setup(rear_left_in2, GPIO.OUT)
GPIO.setup(rear_left_en, GPIO.OUT)
GPIO.output(rear_left_in1, GPIO.LOW)
GPIO.output(rear_left_in2, GPIO.LOW)
rear_left_pwm = GPIO.PWM(rear_left_en, 1000)
rear_left_pwm.start(25)

# Setup rear right wheel
GPIO.setup(rear_right_in1, GPIO.OUT)
GPIO.setup(rear_right_in2, GPIO.OUT)
GPIO.setup(rear_right_en, GPIO.OUT)
GPIO.output(rear_right_in1, GPIO.LOW)
GPIO.output(rear_right_in2, GPIO.LOW)
rear_right_pwm = GPIO.PWM(rear_right_en, 1000)
rear_right_pwm.start(25)

# Default states
front_left_temp = 1
front_right_temp = 1
rear_left_temp = 1
rear_right_temp = 1

print("\n")
print("Four Wheel Robot Control System")
print("Commands:")
print("w - forward  s - stop")
print("a - left     d - right")
print("q - rotate left  e - rotate right")
print("z - backward")
print("1 - low speed  2 - medium speed  3 - high speed")
print("x - exit")
print("\n")


def set_speed(speed):
    """Set speed for all motors"""
    if speed == 1:  # low
        duty_cycle = 25
    elif speed == 2:  # medium
        duty_cycle = 50
    elif speed == 3:  # high
        duty_cycle = 75
    else:
        return

    front_left_pwm.ChangeDutyCycle(duty_cycle)
    front_right_pwm.ChangeDutyCycle(duty_cycle)
    rear_left_pwm.ChangeDutyCycle(duty_cycle)
    rear_right_pwm.ChangeDutyCycle(duty_cycle)
    print(f"Speed set to {'low' if speed == 1 else 'medium' if speed == 2 else 'high'}")


def forward():
    """Move all wheels forward"""
    # Front wheels
    GPIO.output(front_left_in1, GPIO.HIGH)
    GPIO.output(front_left_in2, GPIO.LOW)
    GPIO.output(front_right_in1, GPIO.HIGH)
    GPIO.output(front_right_in2, GPIO.LOW)
    
    # Rear wheels
    GPIO.output(rear_left_in1, GPIO.HIGH)
    GPIO.output(rear_left_in2, GPIO.LOW)
    GPIO.output(rear_right_in1, GPIO.HIGH)
    GPIO.output(rear_right_in2, GPIO.LOW)
    
    print("Moving forward")


def backward():
    """Move all wheels backward"""
    # Front wheels
    GPIO.output(front_left_in1, GPIO.LOW)
    GPIO.output(front_left_in2, GPIO.HIGH)
    GPIO.output(front_right_in1, GPIO.LOW)
    GPIO.output(front_right_in2, GPIO.HIGH)
    
    # Rear wheels
    GPIO.output(rear_left_in1, GPIO.LOW)
    GPIO.output(rear_left_in2, GPIO.HIGH)
    GPIO.output(rear_right_in1, GPIO.LOW)
    GPIO.output(rear_right_in2, GPIO.HIGH)
    
    print("Moving backward")


def stop():
    """Stop all wheels"""
    # Front wheels
    GPIO.output(front_left_in1, GPIO.LOW)
    GPIO.output(front_left_in2, GPIO.LOW)
    GPIO.output(front_right_in1, GPIO.LOW)
    GPIO.output(front_right_in2, GPIO.LOW)
    
    # Rear wheels
    GPIO.output(rear_left_in1, GPIO.LOW)
    GPIO.output(rear_left_in2, GPIO.LOW)
    GPIO.output(rear_right_in1, GPIO.LOW)
    GPIO.output(rear_right_in2, GPIO.LOW)
    
    print("Stopped")


def turn_left():
    """Turn left (right wheels forward, left wheels stopped)"""
    # Front wheels
    GPIO.output(front_left_in1, GPIO.LOW)
    GPIO.output(front_left_in2, GPIO.LOW)
    GPIO.output(front_right_in1, GPIO.HIGH)
    GPIO.output(front_right_in2, GPIO.LOW)
    
    # Rear wheels
    GPIO.output(rear_left_in1, GPIO.LOW)
    GPIO.output(rear_left_in2, GPIO.LOW)
    GPIO.output(rear_right_in1, GPIO.HIGH)
    GPIO.output(rear_right_in2, GPIO.LOW)
    
    print("Turning left")


def turn_right():
    """Turn right (left wheels forward, right wheels stopped)"""
    # Front wheels
    GPIO.output(front_left_in1, GPIO.HIGH)
    GPIO.output(front_left_in2, GPIO.LOW)
    GPIO.output(front_right_in1, GPIO.LOW)
    GPIO.output(front_right_in2, GPIO.LOW)
    
    # Rear wheels
    GPIO.output(rear_left_in1, GPIO.HIGH)
    GPIO.output(rear_left_in2, GPIO.LOW)
    GPIO.output(rear_right_in1, GPIO.LOW)
    GPIO.output(rear_right_in2, GPIO.LOW)
    
    print("Turning right")


def rotate_left():
    """Rotate left (right wheels forward, left wheels backward)"""
    # Front wheels
    GPIO.output(front_left_in1, GPIO.LOW)
    GPIO.output(front_left_in2, GPIO.HIGH)
    GPIO.output(front_right_in1, GPIO.HIGH)
    GPIO.output(front_right_in2, GPIO.LOW)
    
    # Rear wheels
    GPIO.output(rear_left_in1, GPIO.LOW)
    GPIO.output(rear_left_in2, GPIO.HIGH)
    GPIO.output(rear_right_in1, GPIO.HIGH)
    GPIO.output(rear_right_in2, GPIO.LOW)
    
    print("Rotating left")


def rotate_right():
    """Rotate right (left wheels forward, right wheels backward)"""
    # Front wheels
    GPIO.output(front_left_in1, GPIO.HIGH)
    GPIO.output(front_left_in2, GPIO.LOW)
    GPIO.output(front_right_in1, GPIO.LOW)
    GPIO.output(front_right_in2, GPIO.HIGH)
    
    # Rear wheels
    GPIO.output(rear_left_in1, GPIO.HIGH)
    GPIO.output(rear_left_in2, GPIO.LOW)
    GPIO.output(rear_right_in1, GPIO.LOW)
    GPIO.output(rear_right_in2, GPIO.HIGH)
    
    print("Rotating right")


try:
    while True:
        x = input().lower()  # For Python 3

        if x == 'w':
            forward()
        elif x == 's':
            stop()
        elif x == 'a':
            turn_left()
        elif x == 'd':
            turn_right()
        elif x == 'q':
            rotate_left()
        elif x == 'e':
            rotate_right()
        elif x == 'z':
            backward()
        elif x == '1':
            set_speed(1)
        elif x == '2':
            set_speed(2)
        elif x == '3':
            set_speed(3)
        elif x == 'x':
            print("Exiting...")
            break
        else:
            print("Invalid command. Please try again.")

except KeyboardInterrupt:
    print("\nProgram interrupted")

finally:
    # Cleanup GPIO
    front_left_pwm.stop()
    front_right_pwm.stop()
    rear_left_pwm.stop()
    rear_right_pwm.stop()
    GPIO.cleanup()
    print("GPIO cleanup complete")
