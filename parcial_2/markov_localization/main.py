#!/usr/bin/env pybricks-micropython
# Juan Luis Quistian Navarro
"""
    This program implements a markov localization algorithm for a robot
"""
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, ColorSensor, GyroSensor)
from pybricks.parameters import Port, Color
from pybricks.tools import DataLog
from pybricks.robotics import DriveBase
import math
from typing import List

# Directions mapping
DIRECTIONS = {
    'no_move': [0, 0],
    'right': [0, 1],
    'left': [0, -1],
    'down': [1, 0],
    'up': [-1, 0]
}

# Initialize EV3 brick
ev3 = EV3Brick()

# Initialize motors and sensors
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)
color_sensor = ColorSensor(Port.S3)
gyro_sensor = GyroSensor(Port.S2)

# Initialize DriveBase
robot = DriveBase(left_motor, right_motor, wheel_diameter=55.5, axle_track=114)

# Constants
LINE_COLOR = Color.BLACK
BACKGROUND_COLOR = Color.WHITE
LANDMARK_RGB = {
    'red': (71, 69, 83),
    'green': (8, 30, 13),
    'yellow': (74, 82, 71),
    'white': (75, 92, 86),
}
COLOR_THRESHOLD = 50

MOVE_SPEED = 75  # mm/s
DISTANCE_MOVE = 190  # mm

sensor_right = 0.8
sensor_wrong = 1 - sensor_right
p_move = 0.88
p_stay = 1 - p_move

# Mapping angles to directions
DIRECTION_ANGLES = {
    0: -180,  # Up
    1: -90,   # Right
    2: 0,     # Down
    3: 90     # Left
}

# Mapping direction to vectors
DIRECTION_VECTORS = {
    0: DIRECTIONS['up'],
    1: DIRECTIONS['right'],
    2: DIRECTIONS['down'],
    3: DIRECTIONS['left']
}

# grid with size 5x9
world = [
    ['white', 'white', 'white', 'yellow', 'white',
        'white', 'white', 'white', 'white'],
    ['white', 'white', 'white', 'white', 'white',
        'white', 'white', 'yellow', 'white'],
    ['white', 'red', 'white', 'white', 'yellow',
        'white', 'green', 'white', 'white'],
    ['white', 'white', 'green', 'white', 'white',
        'white', 'white', 'white', 'white'],
    ['white', 'white', 'white', 'white', 'white',
        'white', 'white', 'white', 'green']
]

# Data logging
log = DataLog(
    'step', 'gt_x', 'gt_y', 'b_x', 'b_y', 'b_prob', 'measurement',
    'motion_x', 'motion_y', 'color', 'cell_changed', 'direction',
    'current_angle', timestamp=True)


def initialize_probabilities() -> List[List[float]]:
    """Initialize uniform probability distribution"""
    pinit = 1.0 / (len(world) * len(world[0]))
    return [[pinit for _ in range(len(world[0]))] for _ in range(len(world))]


def sense(p: List[List[float]], measurement: str) -> List[List[float]]:
    """Update belief based on sensor measurement"""
    aux = [[0.0 for _ in range(len(p[0]))] for _ in range(len(p))]
    s = 0.0

    for i in range(len(p)):
        for j in range(len(p[0])):
            hit = (measurement == world[i][j])
            aux[i][j] = p[i][j] * (
                hit * sensor_right + (1-hit) * (1-sensor_wrong))
            s += aux[i][j]

    # Normalize
    for i in range(len(aux)):
        for j in range(len(aux[0])):
            aux[i][j] /= s
    return aux


def move(p: List[List[float]], motion: List[int]) -> List[List[float]]:
    """Update belief based on motion"""
    rows, cols = len(p), len(p[0])
    aux = [[0.0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            from_i = i - motion[0]
            from_j = j - motion[1]

            if 0 <= from_i < rows and 0 <= from_j < cols:
                aux[i][j] += p_move * p[from_i][from_j]

            aux[i][j] += p_stay * p[i][j]

    total = sum(sum(row) for row in aux)
    for i in range(rows):
        for j in range(cols):
            aux[i][j] /= total

    return aux


def get_most_probable_pos(p: List[List[float]]) -> tuple[List[int], float]:
    """Get the most probable position from belief state"""
    max_prob = 0.0
    max_pos = [0, 0]
    for i in range(len(p)):
        for j in range(len(p[0])):
            if p[i][j] > max_prob:
                max_prob = p[i][j]
                max_pos = [i, j]
    return max_pos, max_prob


def turn_to_dir(direction: List[int], current_angle: int) -> int:
    """
    Rotate the robot to face the specified direction
    """
    direction_targets = {
        'up': -180,
        'down': 0,
        'right': -90,
        'left': 90
    }

    dir_name = next((k for k, v in DIRECTIONS.items() if v == direction), None)
    if dir_name is None:
        return current_angle

    target_angle = direction_targets.get(dir_name, current_angle)
    TOLERANCE = 5

    angle_diff = (target_angle - current_angle) % 360
    if angle_diff > 180:
        angle_diff -= 360

    turn_direction = 1 if angle_diff > 0 else -1

    robot.drive(0, MOVE_SPEED * turn_direction)

    while True:
        current_angle = gyro_sensor.angle()
        normalized_diff = (target_angle - current_angle) % 360
        if normalized_diff > 180:
            normalized_diff -= 360
        if abs(normalized_diff) < TOLERANCE:
            break

    robot.stop()
    gyro_sensor.reset_angle(target_angle)
    return target_angle


def change_direction(direction: int, current_angle: int) -> int:
    """Change the robot's direction"""
    vector_dir = DIRECTION_VECTORS.get(direction)
    if vector_dir is None:
        return current_angle

    desired_angle = turn_to_dir(vector_dir, current_angle)
    robot.stop()
    robot.straight(-15)
    robot.stop()
    return desired_angle


def move_to_next_cell(current_angle: int) -> None:
    """Move the robot to the next cell in the grid"""
    gyro_sensor.reset_angle(current_angle)
    KP = 1.8
    MAX_DEVIATION = 5

    while color_sensor.color() != LINE_COLOR:
        deviation = gyro_sensor.angle() - current_angle
        steering = -KP * deviation if abs(deviation) >= MAX_DEVIATION else 0
        robot.drive(MOVE_SPEED, steering)

    robot.stop()
    robot.straight(DISTANCE_MOVE)

    current_angle = gyro_sensor.angle()
    angle_diff = (current_angle - gyro_sensor.angle()) % 360
    if angle_diff > 180:
        angle_diff -= 360

    if abs(angle_diff) > 2:
        robot.turn(-angle_diff)
        current_angle = gyro_sensor.angle()


def lawnmower_vertical(current_pos: List[int]) -> int:
    """Generate next movement in vertical lawnmower pattern"""
    max_row = len(world) - 1
    max_col = len(world[0]) - 1
    i, j = current_pos

    # Determine movement direction
    if j % 2 == 0:  # Even columns go down
        return 2 if i < max_row else (1 if j < max_col else None)
    else:  # Odd columns go up
        return 0 if i > 0 else (1 if j < max_col else None)


def lawnmower_vertical_reverse(current_pos: List[int]) -> int:
    """Move in reverse vertical lawnmower pattern"""
    max_row = len(world) - 1
    max_col = len(world[0]) - 1
    i, j = current_pos

    if (max_col - j) % 2 == 0:
        return 0 if i > 0 else (3 if j > 0 else None)
    else:
        return 2 if i < max_row else (3 if j > 0 else None)


def manhattan_move(current_pos: List[int], goal_pos: List[int]) -> int:
    """Generate movement towards goal using Manhattan distance"""
    dx = goal_pos[0] - current_pos[0]
    dy = goal_pos[1] - current_pos[1]

    if abs(dx) > abs(dy):
        return 0 if dx < 0 else 2
    elif dy != 0:
        return 3 if dy < 0 else 1
    return None


def find_goal() -> List[int]:
    """Find goal position in world"""
    for i, row in enumerate(world):
        for j, cell in enumerate(row):
            if cell == 'red':
                return [i, j]
    return None


def rgb_to_color(rgb: tuple) -> str:
    """Convert RGB reading to color name"""
    r, g, b = rgb
    best_match = 'white'
    min_distance = float('inf')

    for color_name, (cr, cg, cb) in LANDMARK_RGB.items():
        distance = math.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        if distance < min_distance and distance < COLOR_THRESHOLD:
            min_distance = distance
            best_match = color_name

    return best_match


def infer_initial_direction(start_pos: List[int]) -> int:
    return 2 if start_pos[1] % 2 == 0 else 0


def get_map_color(color_reading: str | tuple) -> str:
    """Get the color of the cell using RGB readings"""
    if isinstance(color_reading, str):
        return (
            color_reading.lower()
            if color_reading.lower() in LANDMARK_RGB
            else 'white'
        )
    return rgb_to_color(color_reading)


if __name__ == "__main__":
    p = initialize_probabilities()
    gt_pos = [1, 1]
    last_direction = infer_initial_direction(gt_pos)
    current_angle = DIRECTION_ANGLES[last_direction]
    b_pos, b_prob = get_most_probable_pos(p)
    goal_pos = find_goal()

    gt_trajectory = [gt_pos.copy()]
    b_trajectory = [b_pos.copy()]

    step = 0
    found = False
    ev3.screen.print("Starting localization")
    current_lawnmower = lawnmower_vertical

    log.log(
        step, gt_pos[0], gt_pos[1], b_pos[0], b_pos[1],
        b_prob, 'None', 0, 0, str(color_sensor.color()), 0,
        last_direction, current_angle
    )

    while True:
        color = color_sensor.rgb()
        measurement = get_map_color(color)
        ev3.screen.print("m:", measurement)

        p = sense(p, measurement)
        b_pos, b_prob = get_most_probable_pos(p)

        if not found:
            direction = current_lawnmower(gt_pos)
            if direction is None:
                current_lawnmower = (
                    lawnmower_vertical_reverse
                    if current_lawnmower == lawnmower_vertical
                    else lawnmower_vertical
                )
                continue

            # Update ground truth position
            motion = DIRECTION_VECTORS[direction]
            gt_pos[0] = max(0, min(gt_pos[0] + motion[0], len(world)-1))
            gt_pos[1] = max(0, min(gt_pos[1] + motion[1], len(world[0])-1))

            if direction != last_direction:
                current_angle = change_direction(direction, current_angle)
                last_direction = direction
            move_to_next_cell(current_angle)
            p = move(p, motion)

            if b_prob > 0.6:
                found = True
                ev3.speaker.beep()
        else:
            if gt_pos == goal_pos:
                ev3.speaker.beep(frequency=1000, duration=500)
                break

            direction = manhattan_move(gt_pos, goal_pos)
            if direction is None:
                break

            if direction != last_direction:
                current_angle = change_direction(direction, current_angle)
                last_direction = direction
            move_to_next_cell(current_angle)

            motion = DIRECTION_VECTORS[direction]
            gt_pos[0] = max(0, min(gt_pos[0] + motion[0], len(world)-1))
            gt_pos[1] = max(0, min(gt_pos[1] + motion[1], len(world[0])-1))

        gt_trajectory.append(gt_pos.copy())
        b_pos, b_prob = get_most_probable_pos(p)
        b_trajectory.append(b_pos.copy())

        step += 1
        log.log(
            step, gt_pos[0], gt_pos[1], b_pos[0], b_pos[1], b_prob,
            measurement, motion[0], motion[1], str(color_sensor.color()),
            1, direction, current_angle
        )

    ev3.screen.print("Localization complete")
robot.stop()
