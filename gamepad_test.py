import RPi.GPIO as GPIO
import pygame
import cv2
import numpy as np
from time import sleep, time
import os
import threading
import json
from datetime import datetime
import atexit
from gpiozero import MCP3008  # For analog voltage readings

# Set up directory structure for data collection
DATA_DIR = "training_data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
VOLTAGE_DIR = os.path.join(DATA_DIR, "voltage")
CONTROL_DIR = os.path.join(DATA_DIR, "controls")

for directory in [DATA_DIR, IMAGE_DIR, VOLTAGE_DIR, CONTROL_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

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

# Setup MCP3008 ADC for voltage readings (assuming voltage divider on channel 0)
adc = MCP3008(channel=0)

# Initialize camera
camera = cv2.VideoCapture(0)  # Use 0 for default camera
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
if not camera.isOpened():
    print("Error: Could not open camera.")
    GPIO.cleanup()
    exit()

# Initialize pygame for joystick
pygame.init()
pygame.joystick.init()

# Check if any joystick/controller is connected
if pygame.joystick.get_count() == 0:
    print("Error: No joystick/controller found.")
    GPIO.cleanup()
    camera.release()
    pygame.quit()
    exit()

# Initialize the first joystick
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Controller initialized: {joystick.get_name()}")

# Default states and variables
left_temp = 1
right_temp = 1
current_speed = 2  # Default to medium speed
speed_map = {1: 25, 2: 50, 3: 75}  # Speed levels
recording = False  # Flag to toggle data recording
record_interval = 0.1  # Record data every 100ms
last_record_time = 0
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
frame_counter = 0

# Xbox controller mapping (may vary slightly depending on controller model)
AXIS_LEFT_STICK_X = 0
AXIS_LEFT_STICK_Y = 1
AXIS_RIGHT_STICK_X = 4
AXIS_RIGHT_STICK_Y = 3
AXIS_TRIGGER_LEFT = 2
AXIS_TRIGGER_RIGHT = 5
BUTTON_A = 0
BUTTON_B = 1
BUTTON_X = 2
BUTTON_Y = 3
BUTTON_LB = 4
BUTTON_RB = 5
BUTTON_BACK = 6
BUTTON_START = 7
BUTTON_LEFT_STICK = 8
BUTTON_RIGHT_STICK = 9

# Dead zone for joystick
DEAD_ZONE = 0.15


def read_voltage():
    """Read voltage from MCP3008 ADC"""
    # Assuming a voltage divider with R1=10k and R2=1k for readings up to 11V
    # Scale the ADC reading (0-1) to voltage
    raw_value = adc.value
    # Calculate actual voltage based on your voltage divider
    actual_voltage = raw_value * 11.0  # Adjust multiplier based on your setup
    return actual_voltage


def set_speed(speed_level):
    """Set speed for both motors"""
    global current_speed

    if speed_level in speed_map:
        duty_cycle = speed_map[speed_level]
        left_pwm.ChangeDutyCycle(duty_cycle)
        right_pwm.ChangeDutyCycle(duty_cycle)
        current_speed = speed_level
        print(f"Speed set to {'low' if speed_level == 1 else 'medium' if speed_level == 2 else 'high'}")


def set_motor_speeds(left_speed, right_speed):
    """Set individual motor speeds based on normalized inputs (-1 to 1)"""
    # Convert from -1:1 to 0:100 for PWM
    left_duty = abs(left_speed) * speed_map[current_speed]
    right_duty = abs(right_speed) * speed_map[current_speed]

    # Set direction based on sign
    if left_speed > 0:
        GPIO.output(left_in1, GPIO.HIGH)
        GPIO.output(left_in2, GPIO.LOW)
    elif left_speed < 0:
        GPIO.output(left_in1, GPIO.LOW)
        GPIO.output(left_in2, GPIO.HIGH)
    else:
        GPIO.output(left_in1, GPIO.LOW)
        GPIO.output(left_in2, GPIO.LOW)

    if right_speed > 0:
        GPIO.output(right_in1, GPIO.HIGH)
        GPIO.output(right_in2, GPIO.LOW)
    elif right_speed < 0:
        GPIO.output(right_in1, GPIO.LOW)
        GPIO.output(right_in2, GPIO.HIGH)
    else:
        GPIO.output(right_in1, GPIO.LOW)
        GPIO.output(right_in2, GPIO.LOW)

    # Apply the duty cycle
    left_pwm.ChangeDutyCycle(left_duty)
    right_pwm.ChangeDutyCycle(right_duty)


def toggle_recording():
    """Toggle data recording state"""
    global recording
    recording = not recording
    print(f"Recording {'started' if recording else 'stopped'}")


def save_image(frame):
    """Save camera frame to disk"""
    global frame_counter
    filename = os.path.join(IMAGE_DIR, f"{session_id}_{frame_counter:06d}.jpg")
    cv2.imwrite(filename, frame)
    return filename


def save_voltage(voltage):
    """Save voltage reading to disk"""
    global frame_counter
    filename = os.path.join(VOLTAGE_DIR, f"{session_id}_{frame_counter:06d}.json")
    with open(filename, 'w') as f:
        json.dump({"voltage": voltage, "timestamp": time()}, f)
    return filename


def save_control_data(left_speed, right_speed):
    """Save control inputs to disk"""
    global frame_counter
    filename = os.path.join(CONTROL_DIR, f"{session_id}_{frame_counter:06d}.json")
    with open(filename, 'w') as f:
        json.dump({
            "left_speed": left_speed,
            "right_speed": right_speed,
            "timestamp": time()
        }, f)
    return filename


def record_data(left_speed, right_speed):
    """Record a complete set of data (image, voltage, controls)"""
    global frame_counter, last_record_time

    # Check if it's time to record
    current_time = time()
    if current_time - last_record_time < record_interval:
        return

    # Capture frame
    ret, frame = camera.read()
    if not ret:
        print("Error: Couldn't capture frame")
        return

    # Read voltage
    voltage = read_voltage()

    # Save all data
    image_file = save_image(frame)
    voltage_file = save_voltage(voltage)
    control_file = save_control_data(left_speed, right_speed)

    frame_counter += 1
    last_record_time = current_time

    # Occasionally print status
    if frame_counter % 10 == 0:
        print(f"Recorded frame #{frame_counter}, voltage: {voltage:.2f}V")


def stop():
    """Stop both wheels"""
    GPIO.output(left_in1, GPIO.LOW)
    GPIO.output(left_in2, GPIO.LOW)
    GPIO.output(right_in1, GPIO.LOW)
    GPIO.output(right_in2, GPIO.LOW)
    print("Stopped")


def cleanup():
    """Clean up resources on exit"""
    print("Cleaning up...")
    stop()
    left_pwm.stop()
    right_pwm.stop()
    GPIO.cleanup()
    camera.release()
    pygame.quit()
    print("Cleanup complete")


# Register cleanup to happen at exit
atexit.register(cleanup)

print("\n")
print("Xbox Controller Robot Control System")
print("-----------------------------------")
print("Left stick: Drive robot (tank-style control)")
print("Right trigger: Increase speed")
print("Left trigger: Decrease speed")
print("A button: Toggle data recording")
print("B button: Emergency stop")
print("Back button: Exit program")
print("\n")

try:
    running = True
    while running:
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if joystick.get_button(BUTTON_A):
                    toggle_recording()
                elif joystick.get_button(BUTTON_B):
                    stop()
                elif joystick.get_button(BUTTON_BACK):
                    print("Exiting...")
                    running = False
                    break

        # Get joystick values
        left_stick_y = -joystick.get_axis(AXIS_LEFT_STICK_Y)  # Invert Y axis
        left_stick_x = joystick.get_axis(AXIS_LEFT_STICK_X)

        # Apply deadzone
        if abs(left_stick_y) < DEAD_ZONE:
            left_stick_y = 0
        if abs(left_stick_x) < DEAD_ZONE:
            left_stick_x = 0

        # Check triggers for speed control
        right_trigger = (joystick.get_axis(AXIS_TRIGGER_RIGHT) + 1) / 2  # Convert -1:1 to 0:1
        left_trigger = (joystick.get_axis(AXIS_TRIGGER_LEFT) + 1) / 2  # Convert -1:1 to 0:1

        if right_trigger > 0.8 and current_speed < 3:
            set_speed(current_speed + 1)
            sleep(0.3)  # Debounce
        elif left_trigger > 0.8 and current_speed > 1:
            set_speed(current_speed - 1)
            sleep(0.3)  # Debounce

        # Calculate motor speeds using tank-style control
        # Mix forward/backward with turning
        left_speed = 0
        right_speed = 0

        # Pure differential drive calculation
        if abs(left_stick_y) > abs(left_stick_x):
            # Primarily forward/backward motion
            left_speed = left_stick_y - left_stick_x
            right_speed = left_stick_y + left_stick_x
        else:
            # Primarily turning motion
            left_speed = left_stick_y - left_stick_x
            right_speed = left_stick_y + left_stick_x

        # Normalize speeds to range -1 to 1
        max_value = max(1.0, abs(left_speed), abs(right_speed))
        left_speed = left_speed / max_value
        right_speed = right_speed / max_value

        # Apply motor speeds
        set_motor_speeds(left_speed, right_speed)

        # Record data if enabled
        if recording:
            record_data(left_speed, right_speed)

        # Small sleep to prevent CPU hogging
        sleep(0.01)

except KeyboardInterrupt:
    print("\nProgram interrupted by user")
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    cleanup()