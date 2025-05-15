import RPi.GPIO as GPIO
import pygame
from time import sleep

# Front wheels GPIO pins
front_left_in1 = 20
front_left_in2 = 21
front_left_en = 16

front_right_in1 = 13
front_right_in2 = 26
front_right_en = 19

# Rear wheels GPIO pins
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

# Initialize pygame and joystick
pygame.init()
pygame.joystick.init()

# Speed and movement variables
speed_level = 1  # 1=low, 2=medium, 3=high
current_speed = 25  # PWM duty cycle


def set_speed(speed):
    """Set speed for all motors"""
    global speed_level, current_speed
    
    if speed == 1:  # low
        duty_cycle = 25
    elif speed == 2:  # medium
        duty_cycle = 50
    elif speed == 3:  # high
        duty_cycle = 75
    else:
        return
    
    speed_level = speed
    current_speed = duty_cycle
    
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
    """Turn left (right wheels forward, left wheels backward)"""
    # Front wheels
    GPIO.output(front_left_in1, GPIO.LOW)
    GPIO.output(front_left_in2, GPIO.HIGH)  # Changed to backward
    GPIO.output(front_right_in1, GPIO.HIGH)
    GPIO.output(front_right_in2, GPIO.LOW)
    
    # Rear wheels
    GPIO.output(rear_left_in1, GPIO.HIGH)
    GPIO.output(rear_left_in2, GPIO.LOW)  # Changed to backward
    GPIO.output(rear_right_in1, GPIO.LOW)
    GPIO.output(rear_right_in2, GPIO.HIGH)
    
    print("Turning left")


def turn_right():
    """Turn right (left wheels forward, right wheels backward)"""
    # Front wheels
    GPIO.output(front_left_in1, GPIO.HIGH)
    GPIO.output(front_left_in2, GPIO.LOW)
    GPIO.output(front_right_in1, GPIO.LOW)
    GPIO.output(front_right_in2, GPIO.HIGH)  # Changed to backward
    
    # Rear wheels
    GPIO.output(rear_left_in1, GPIO.LOW)
    GPIO.output(rear_left_in2, GPIO.HIGH)
    GPIO.output(rear_right_in1, GPIO.HIGH)
    GPIO.output(rear_right_in2, GPIO.LOW)  # Changed to backward
    
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


def handle_joystick_movement(x_axis, y_axis):
    """Handle joystick movement for robot control"""
    # Introduce deadzone to prevent minor joystick movements
    deadzone = 0.2
    
    if abs(x_axis) < deadzone and abs(y_axis) < deadzone:
        stop()
        return
    
    # Forward/backward control (y-axis)
    if y_axis < -deadzone:  # Push up on stick
        if abs(x_axis) > deadzone:
            if x_axis < 0:  # Push left
                turn_left()
            else:  # Push right
                turn_right()
        else:
            forward()
    elif y_axis > deadzone:  # Push down on stick
        backward()
    # Pure left/right rotation when x-axis is moved but y-axis is near center
    elif abs(x_axis) > deadzone and abs(y_axis) <= deadzone:
        if x_axis < 0:  # Push left
            rotate_left()
        else:  # Push right
            rotate_right()


def print_instructions():
    """Print gamepad control instructions"""
    print("\n")
    print("Four Wheel Robot Control System - Gamepad Edition")
    print("-----------------------------------------------")
    print("Controls:")
    print("Left analog stick: Movement control")
    print("  - Up: Forward")
    print("  - Down: Backward")
    print("  - Left: Turn left")
    print("  - Right: Turn right")
    print("  - Diagonal left/right: Steering")
    print("X/A: Speed level 1 (low)")
    print("Y/B: Speed level 2 (medium)")
    print("B/X: Speed level 3 (high)")
    print("Start: Exit program")
    print("\n")
    print("Waiting for gamepad input...")


try:
    # Check for joysticks
    joystick_count = pygame.joystick.get_count()
    
    if joystick_count == 0:
        print("No gamepad detected. Please connect a gamepad and restart.")
        # Fall back to keyboard control
        print("Falling back to keyboard control.")
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
        
        keyboard_mode = True
    else:
        # Use the first joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        
        print(f"Detected gamepad: {joystick.get_name()}")
        print_instructions()
        keyboard_mode = False
    
    running = True
    
    while running:
        if keyboard_mode:
            # Keyboard control
            x = input().lower()
            
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
                running = False
            else:
                print("Invalid command. Please try again.")
        else:
            # Gamepad control
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    # Button mappings may vary by controller - adjust as needed
                    if event.button == 0:  # A/X button (usually)
                        set_speed(1)
                    elif event.button == 1:  # B/Y button (usually)
                        set_speed(2)
                    elif event.button == 2:  # X/B button (usually)
                        set_speed(3)
                    elif event.button == 7:  # Start button (usually)
                        print("Exiting...")
                        running = False
            
            if joystick_count > 0 and running:
                # Read left analog stick
                x_axis = joystick.get_axis(0)  # Left/Right
                y_axis = joystick.get_axis(1)  # Up/Down
                
                handle_joystick_movement(x_axis, y_axis)
            
            # Small delay to prevent CPU overuse
            sleep(0.1)

except KeyboardInterrupt:
    print("\nProgram interrupted")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Cleanup
    if 'pygame' in locals() or 'pygame' in globals():
        pygame.quit()
    
    # Cleanup GPIO
    front_left_pwm.stop()
    front_right_pwm.stop()
    rear_left_pwm.stop()
    rear_right_pwm.stop()
    GPIO.cleanup()
    print("GPIO cleanup complete")
