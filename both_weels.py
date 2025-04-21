import RPi.GPIO as GPIO
from time import sleep

# Left wheel GPIO pins
left_in1 = 24
left_in2 = 23
left_en = 25

# Right wheel GPIO pins
right_in1 = 7
right_in2 = 8
right_en = 12

# Initialize GPIO
GPIO.setmode(GPIO.BCM)

# Setup left wheel
GPIO.setup(left_in1, GPIO.OUT)
GPIO.setup(left_in2, GPIO.OUT)
GPIO.setup(left_en, GPIO.OUT)
GPIO.output(left_in1, GPIO.LOW)
GPIO.output(left_in2, GPIO.LOW)
left_pwm = GPIO.PWM(left_en, 1000)
left_pwm.start(25)

# Setup right wheel
GPIO.setup(right_in1, GPIO.OUT)
GPIO.setup(right_in2, GPIO.OUT)
GPIO.setup(right_en, GPIO.OUT)
GPIO.output(right_in1, GPIO.LOW)
GPIO.output(right_in2, GPIO.LOW)
right_pwm = GPIO.PWM(right_en, 1000)
right_pwm.start(25)

# Default states
left_temp = 1
right_temp = 1

print("\n")
print("Dual Motor Control System")
print("Commands:")
print("w - forward  s - stop")
print("a - left     d - right")
print("q - rotate left  e - rotate right")
print("z - backward")
print("1 - low speed  2 - medium speed  3 - high speed")
print("x - exit")
print("\n")

def set_speed(speed):
    """Set speed for both motors"""
    if speed == 1:  # low
        duty_cycle = 25
    elif speed == 2:  # medium
        duty_cycle = 50
    elif speed == 3:  # high
        duty_cycle = 75
    else:
        return
    
    left_pwm.ChangeDutyCycle(duty_cycle)
    right_pwm.ChangeDutyCycle(duty_cycle)
    print(f"Speed set to {'low' if speed == 1 else 'medium' if speed == 2 else 'high'}")

def forward():
    """Move both wheels forward"""
    GPIO.output(left_in1, GPIO.HIGH)
    GPIO.output(left_in2, GPIO.LOW)
    GPIO.output(right_in1, GPIO.HIGH)
    GPIO.output(right_in2, GPIO.LOW)
    left_temp = 1
    right_temp = 1
    print("Moving forward")

def backward():
    """Move both wheels backward"""
    GPIO.output(left_in1, GPIO.LOW)
    GPIO.output(left_in2, GPIO.HIGH)
    GPIO.output(right_in1, GPIO.LOW)
    GPIO.output(right_in2, GPIO.HIGH)
    left_temp = 0
    right_temp = 0
    print("Moving backward")

def stop():
    """Stop both wheels"""
    GPIO.output(left_in1, GPIO.LOW)
    GPIO.output(left_in2, GPIO.LOW)
    GPIO.output(right_in1, GPIO.LOW)
    GPIO.output(right_in2, GPIO.LOW)
    print("Stopped")

def turn_left():
    """Turn left (right wheel forward, left wheel stopped)"""
    GPIO.output(left_in1, GPIO.LOW)
    GPIO.output(left_in2, GPIO.LOW)
    GPIO.output(right_in1, GPIO.HIGH)
    GPIO.output(right_in2, GPIO.LOW)
    print("Turning left")

def turn_right():
    """Turn right (left wheel forward, right wheel stopped)"""
    GPIO.output(left_in1, GPIO.HIGH)
    GPIO.output(left_in2, GPIO.LOW)
    GPIO.output(right_in1, GPIO.LOW)
    GPIO.output(right_in2, GPIO.LOW)
    print("Turning right")

def rotate_left():
    """Rotate left (right wheel forward, left wheel backward)"""
    GPIO.output(left_in1, GPIO.LOW)
    GPIO.output(left_in2, GPIO.HIGH)
    GPIO.output(right_in1, GPIO.HIGH)
    GPIO.output(right_in2, GPIO.LOW)
    print("Rotating left")

def rotate_right():
    """Rotate right (left wheel forward, right wheel backward)"""
    GPIO.output(left_in1, GPIO.HIGH)
    GPIO.output(left_in2, GPIO.LOW)
    GPIO.output(right_in1, GPIO.LOW)
    GPIO.output(right_in2, GPIO.HIGH)
    print("Rotating right")

try:
    while True:
        x = input().lower()  # Changed from raw_input() to input() for Python 3
        
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
    left_pwm.stop()
    right_pwm.stop()
    GPIO.cleanup()
    print("GPIO cleanup complete")
