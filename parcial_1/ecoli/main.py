#!/usr/bin/env pybricks-micropython
import threading
import random
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, ColorSensor, GyroSensor)
from pybricks.parameters import Port
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase


class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, X_init=0):
        self.x = X_init  # Estimate
        self.P = 1  # Variance of error
        self.Q = process_variance  # Noise of the process
        self.R = measurement_variance  # Noise of the measurement
        self.K = 0  # Kalman gain

    def update(self, measurement):
        self.P = self.P + self.Q
        self.K = self.P / (self.P + self.R)
        self.x = self.x + self.K * (measurement - self.x)
        self.P = (1 - self.K) * self.P
        return self.x


class LeakyIntegrator:
    def __init__(self, a, b, threshold, X_init):
        self.a = a  # Decay rate
        self.b = b  # Sensitivity to changes
        self.threshold = threshold  # Threshold for behavior change
        self.V = X_init  # Internal state

    def update(self, X):
        self.V = self.V + self.b * X - self.a * self.V
        return self.V

    def get_behavior(self, X):
        U = X - self.V
        if U > self.threshold:
            return "run"  # Move straight
        else:
            return "tumble"  # Change direction


def get_concentration(base):
    r, g, b = color_sensor.rgb()
    return 100 - (r + g + b) / 3 - base


def log_data():
    global stop_logging
    stopwatch = StopWatch()
    last_log_time = 0
    while not stop_logging:
        current_time = stopwatch.time()
        if current_time - last_log_time >= 100:
            distance_traveled = robot.distance()
            gyro_angle = gyro_sensor.angle()
            left_motor_deg = left_motor.angle()
            right_motor_deg = right_motor.angle()
            concentration = get_concentration(0)

            # Apply Kalman filter to reflection and concentration
            kalman_concentration_green_color = kalman_filter_concentration.\
                update(concentration)

            V = leaky_integrator.V
            U = concentration - V

            behavior = leaky_integrator.get_behavior(concentration)
            rounded_time = round(current_time / 100) * 0.1
            log.log(
                rounded_time,
                distance_traveled,
                gyro_angle,
                left_motor_deg,
                right_motor_deg,
                concentration,
                kalman_concentration_green_color,
                V,
                U,
                behavior
            )
            last_log_time = current_time
        wait(10)


def run(duration):
    robot.drive(STRAIGHT_SPEED, 0)
    wait(duration)
    robot.stop()


def tumble():
    robot.turn(random.uniform(-180, 180))


def main_loop():
    global run_robot
    run_robot = True
    times_to_mesure_fisrt_concentration = 10

    # Initial concentration
    for _ in range(times_to_mesure_fisrt_concentration):
        concentration = get_concentration(0)
        kalman_concentration = kalman_filter_concentration\
            .update(concentration)
        V = leaky_integrator.update(kalman_concentration)
        U = kalman_concentration - V
        behavior = leaky_integrator.get_behavior(kalman_concentration)
        ev3.screen.print("U:", round(U, 1), "B:", behavior)
        wait(200)

    while run_robot:
        concentration = get_concentration(0)
        kalman_concentration = kalman_filter_concentration.\
            update(concentration)

        V = leaky_integrator.update(kalman_concentration)
        U = kalman_concentration - V
        behavior = leaky_integrator.get_behavior(kalman_concentration)

        if behavior == "run":
            run(RUN_DURATION)
        else:
            tumble()
            run(TUMBLE_DURATION)

        ev3.screen.print("U:", round(U, 1), "B:", behavior)


# Initialize EV3 brick
ev3 = EV3Brick()

# Initialize motors and sensors
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)
gyro_sensor = GyroSensor(Port.S2)
color_sensor = ColorSensor(Port.S3)

# Initialize DriveBase
robot = DriveBase(left_motor, right_motor, wheel_diameter=56, axle_track=114)

# Initialize DataLog
log = DataLog(
    "timestamp",
    "distance",
    "gyro_angle",
    "left_motor_deg",
    "right_motor_deg",
    "concentration",
    "kalman_concentration",
    "V",
    "U(t)",
    "behavior",
    name="ev3_ecoli_simulation"
    )

# Global variables
stop_logging = False
run_robot = False

# Constants
STRAIGHT_SPEED = 100
TUMBLE_SPEED = 100
RUN_DURATION = 250
TUMBLE_DURATION = 100

# Instances
kalman_filter_concentration = KalmanFilter(process_variance=0.1,
                                           measurement_variance=5,
                                           X_init=int(get_concentration(0)))
leaky_integrator = LeakyIntegrator(a=0.5,
                                   b=0.5,
                                   threshold=0.5,
                                   X_init=get_concentration(0))

if __name__ == "__main__":
    ev3.screen.print("Starting E. coli simulation...")

    # Reset robot and gyro sensor
    robot.reset()
    gyro_sensor.reset_angle(0)

    # Set motor run settings
    left_motor.set_run_settings(100, 200)
    right_motor.set_run_settings(100, 200)

    # thread for logging data
    log_thread = threading.Thread(target=log_data)
    log_thread.start()

    # Start main loop
    main_loop()

    # Stop logging
    stop_logging = True
    run_robot = False

    # Wait for logging to finish
    log_thread.join()

    ev3.screen.print("Simulation complete.")
