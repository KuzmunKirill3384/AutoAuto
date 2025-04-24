import RPi.GPIO as GPIO
import pygame
import numpy as np
from time import sleep, time
import os
import json
from datetime import datetime
import atexit
from gpiozero import MCP3008
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

# Disable GPIO warnings
GPIO.setwarnings(False)

# Set up directory structure
DATA_DIR = "training_data"
VOLTAGE_DIR = os.path.join(DATA_DIR, "voltage")
CONTROL_DIR = os.path.join(DATA_DIR, "controls")
VIDEO_DIR = os.path.join(DATA_DIR, "video")

for directory in [DATA_DIR, VOLTAGE_DIR, CONTROL_DIR, VIDEO_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# GPIO pin setup - Left side motors
left_front_in1, left_front_in2, left_front_en = 24, 23, 25  # From gamepad_test_cross.py
left_rear_in1, left_rear_in2, left_rear_en = 5, 22, 6     # From long_provoda_test.py

# GPIO pin setup - Right side motors
right_front_in1, right_front_in2, right_front_en = 7, 8, 12  # From gamepad_test_cross.py
right_rear_in1, right_rear_in2, right_rear_en = 27, 17, 4    # From long_provoda_test.py

# Initialize GPIO
GPIO.setmode(GPIO.BCM)

# Setup motor pins
motor_pins = [
    left_front_in1, left_front_in2, left_front_en,
    left_rear_in1, left_rear_in2, left_rear_en,
    right_front_in1, right_front_in2, right_front_en,
    right_rear_in1, right_rear_in2, right_rear_en
]

for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Setup PWM
left_front_pwm = GPIO.PWM(left_front_en, 1000)
left_rear_pwm = GPIO.PWM(left_rear_en, 1000)
right_front_pwm = GPIO.PWM(right_front_en, 1000)
right_rear_pwm = GPIO.PWM(right_rear_en, 1000)

# Start PWM at 25% duty cycle
left_front_pwm.start(25)
left_rear_pwm.start(25)
right_front_pwm.start(25)
right_rear_pwm.start(25)

# Setup ADC
try:
    adc = MCP3008(channel=0)
    voltage_sensor_enabled = True
except:
    print("Warning: Could not initialize MCP3008. Voltage readings disabled.")
    voltage_sensor_enabled = False

# Check for camera
camera_enabled = False
camera = None
if Picamera2 is not None:
    try:
        camera = Picamera2()
        camera.configure(camera.create_video_configuration(main={"size": (640, 480)}))
        camera_enabled = True
        print("Camera detected and initialized.")
    except Exception as e:
        print(f"Camera initialization failed: {e}")
        camera = None
else:
    print("Picamera2 module not found.")

if not camera_enabled:
    choice = input("No camera detected. Continue without camera? (y/n): ").lower()
    if choice != 'y':
        print("Exiting due to no camera.")
        exit()

# Initialize pygame
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("Error: No controller found.")
    if camera_enabled:
        camera.close()
    GPIO.cleanup()
    pygame.quit()
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Controller: {joystick.get_name()}")

# Controller mapping
BUTTON_A, BUTTON_B, BUTTON_X, BUTTON_Y = 0, 1, 2, 3
BUTTON_LB, BUTTON_RB = 4, 5
BUTTON_BACK, BUTTON_START = 6, 7
AXIS_TRIGGER_LEFT, AXIS_TRIGGER_RIGHT = 2, 5
HAT_DPAD = 0

# Variables
current_speed = 3
speed_map = {1: 10, 2: 25, 3: 40, 4: 60, 5: 80, 6: 100}
recording = False
record_interval = 0.1
last_record_time = 0
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
frame_counter = 0
video_recording = False

def display_instructions():
    print("\n=== Four-Wheel Robot Control Instructions ===")
    print("D-pad Up/Down: Move forward/backward")
    print("D-pad Left/Right: Turn left/right")
    print("Right Trigger: Increase speed")
    print("Left Trigger: Decrease speed")
    print(f"Speed Levels: {len(speed_map)} (from {speed_map[1]}% to {speed_map[len(speed_map)]}%)")
    print("A Button: Toggle data recording")
    print("B Button: Emergency stop")
    print("Back Button: Exit program")
    if camera_enabled:
        print("Camera: Recording video when data recording is active")
    print("=========================================\n")
    input("Press Enter to start controlling the robot...")

def read_voltage():
    if voltage_sensor_enabled:
        try:
            return adc.value * 11.0
        except:
            return 0.0
    return 0.0

def set_speed(speed_level):
    global current_speed
    if speed_level in speed_map:
        duty_cycle = speed_map[speed_level]
        left_front_pwm.ChangeDutyCycle(duty_cycle)
        left_rear_pwm.ChangeDutyCycle(duty_cycle)
        right_front_pwm.ChangeDutyCycle(duty_cycle)
        right_rear_pwm.ChangeDutyCycle(duty_cycle)
        current_speed = speed_level
        print(f"Speed level: {speed_level} ({duty_cycle}%)")

def set_motor_speeds(left_speed, right_speed):
    left_duty = abs(left_speed) * speed_map[current_speed]
    right_duty = abs(right_speed) * speed_map[current_speed]
    
    # Set left front motor direction
    GPIO.output(left_front_in1, GPIO.HIGH if left_speed > 0 else GPIO.LOW)
    GPIO.output(left_front_in2, GPIO.LOW if left_speed > 0 else GPIO.HIGH if left_speed < 0 else GPIO.LOW)
    
    # Set left rear motor direction
    GPIO.output(left_rear_in1, GPIO.HIGH if left_speed > 0 else GPIO.LOW)
    GPIO.output(left_rear_in2, GPIO.LOW if left_speed > 0 else GPIO.HIGH if left_speed < 0 else GPIO.LOW)
    
    # Set right front motor direction
    GPIO.output(right_front_in1, GPIO.HIGH if right_speed > 0 else GPIO.LOW)
    GPIO.output(right_front_in2, GPIO.LOW if right_speed > 0 else GPIO.HIGH if right_speed < 0 else GPIO.LOW)
    
    # Set right rear motor direction
    GPIO.output(right_rear_in1, GPIO.HIGH if right_speed > 0 else GPIO.LOW)
    GPIO.output(right_rear_in2, GPIO.LOW if right_speed > 0 else GPIO.HIGH if right_speed < 0 else GPIO.LOW)
    
    # Apply PWM duty cycle
    left_front_pwm.ChangeDutyCycle(left_duty)
    left_rear_pwm.ChangeDutyCycle(left_duty)
    right_front_pwm.ChangeDutyCycle(right_duty)
    right_rear_pwm.ChangeDutyCycle(right_duty)

def toggle_recording():
    global recording, video_recording
    recording = not recording
    if camera_enabled:
        if recording and not video_recording:
            video_filename = os.path.join(VIDEO_DIR, f"{session_id}.h264")
            camera.start_recording(video_filename)
            video_recording = True
            print(f"Started video recording: {video_filename}")
        elif not recording and video_recording:
            camera.stop_recording()
            video_recording = False
            print("Stopped video recording")
    print(f"Data recording {'started' if recording else 'stopped'}")

def save_voltage(voltage):
    global frame_counter
    filename = os.path.join(VOLTAGE_DIR, f"{session_id}_{frame_counter:06d}.json")
    with open(filename, 'w') as f:
        json.dump({"voltage": voltage, "timestamp": time()}, f)
    return filename

def save_control_data(left_speed, right_speed):
    global frame_counter
    filename = os.path.join(CONTROL_DIR, f"{session_id}_{frame_counter:06d}.json")
    with open(filename, 'w') as f:
        json.dump({"left_speed": left_speed, "right_speed": right_speed, "timestamp": time()}, f)
    return filename

def record_data(left_speed, right_speed):
    global frame_counter, last_record_time
    current_time = time()
    if current_time - last_record_time < record_interval:
        return

    voltage = read_voltage()
    if voltage_sensor_enabled:
        save_voltage(voltage)
    save_control_data(left_speed, right_speed)

    frame_counter += 1
    last_record_time = current_time
    if frame_counter % 10 == 0:
        status = f"Recorded data #{frame_counter}"
        if voltage_sensor_enabled:
            status += f", voltage: {voltage:.2f}V"
        print(status)

def stop():
    all_in_pins = [
        left_front_in1, left_front_in2,
        left_rear_in1, left_rear_in2,
        right_front_in1, right_front_in2,
        right_rear_in1, right_rear_in2
    ]
    for pin in all_in_pins:
        GPIO.output(pin, GPIO.LOW)
    print("All motors stopped")

def cleanup():
    print("Cleaning up...")
    stop()
    
    # Stop all PWMs
    left_front_pwm.stop()
    left_rear_pwm.stop()
    right_front_pwm.stop()
    right_rear_pwm.stop()
    
    # Handle camera
    if camera_enabled and video_recording:
        camera.stop_recording()
    if camera_enabled:
        camera.close()
        
    GPIO.cleanup()
    pygame.quit()
    print("Cleanup complete")

atexit.register(cleanup)

# Show instructions
display_instructions()

try:
    running = True
    while running:
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

        # Get D-pad values
        dpad = joystick.get_hat(HAT_DPAD)
        dpad_x, dpad_y = dpad[0], dpad[1]

        # Check triggers
        right_trigger = (joystick.get_axis(AXIS_TRIGGER_RIGHT) + 1) / 2
        left_trigger = (joystick.get_axis(AXIS_TRIGGER_LEFT) + 1) / 2

        if right_trigger > 0.8 and current_speed < len(speed_map):
            set_speed(current_speed + 1)
            sleep(0.3)
        elif left_trigger > 0.8 and current_speed > 1:
            set_speed(current_speed - 1)
            sleep(0.3)

        # Calculate motor speeds using D-pad
        left_speed = right_speed = 0
        if dpad_y != 0:
            left_speed = right_speed = dpad_y
        if dpad_x != 0:
            left_speed -= dpad_x
            right_speed += dpad_x

        # Normalize speeds
        max_value = max(1.0, abs(left_speed), abs(right_speed))
        left_speed /= max_value
        right_speed /= max_value

        set_motor_speeds(left_speed, right_speed)

        if recording:
            record_data(left_speed, right_speed)

        sleep(0.01)

except KeyboardInterrupt:
    print("\nProgram interrupted by user")
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    cleanup()
