import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

#   [0, 0] - No move
#   [0, 1] - Right
#   [0, -1] - Left
#   [1, 0] - Down
#   [-1, 0] - Up


def sense(p, world, measurement, sensor_right, sensor_wrong):
    """
    Update the belief state based on the measurement and sensor accuracy.
    """
    aux = [[0.0 for row in range(len(p[0]))] for col in range(len(p))]
    s = 0.0
    for i in range(len(p)):
        for j in range(len(p[i])):
            hit = (measurement == world[i][j])
            aux[i][j] = p[i][j] * (hit * sensor_right + (1-hit) * sensor_wrong)
            s += aux[i][j]
    for i in range(len(aux)):
        for j in range(len(p[i])):
            aux[i][j] /= s
    return aux


def move(p, motion, p_move, p_stay):
    """

    """
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


def show(p):
    """
    Show the probability distribution in a grid format.
    """
    for i in range(len(p)):
        for j in range(len(p[i])):
            print(f"{p[i][j]:.2f} |", end=" ")
        print("\n"+"-"*62)


def find_goal(world):
    """
    Find the position of the goal in the world.
    """
    for i in range(len(world)):
        for j in range(len(world[0])):
            if world[i][j] == 'meta':
                return [i, j]
    return None


def get_measurement(pos_x, pos_y):
    """
    Get the measurement from the world based on the robot's position.
    """

    return world[pos_x][pos_y]


def get_most_probable_pos(p):
    """
    Get the most probable position from the belief state.
    """
    max_prob = 0
    max_pos = [0, 0]
    for i in range(len(p)):
        for j in range(len(p[i])):
            if p[i][j] > max_prob:
                max_prob = p[i][j]
                max_pos = [i, j]
    return max_pos, max_prob


def update_real_postion(gt_pos, motion):
    """
    Update the ground truth position based on the motion vector.
    """
    gt_pos[0] += motion[0]
    gt_pos[1] += motion[1]
    gt_pos[0] = max(0, min(gt_pos[0], len(world)-1))
    gt_pos[1] = max(0, min(gt_pos[1], len(world[0])-1))

    return gt_pos


def lawnmower_horizontal(current_pos):
    """
    Move the robot in a horizontal lawnmower pattern.
    """
    max_row = len(world) - 1
    max_col = len(world[0]) - 1
    i, j = current_pos

    if i % 2 == 0:  # Even rows go right
        if j < max_col:
            return [0, 1]  # Right
        else:
            if i < max_row:
                return [1, 0]  # Down
            else:
                return None  # Reached end, switch to backward
    else:  # Odd rows go left
        if j > 0:
            return [0, -1]  # Left
        else:
            if i < max_row:
                return [1, 0]  # Down
            else:
                return None  # Reached end, switch to backward


def lawnmower_horizontal_reverse(current_pos):
    """
    Move the robot in a horizontal lawnmower pattern in reverse.
    """
    max_row = len(world) - 1
    max_col = len(world[0]) - 1
    i, j = current_pos

    if (max_row - i) % 2 == 0:  # Even reverse rows go left
        if j > 0:
            return [0, -1]  # Left
        else:
            if i > 0:
                return [-1, 0]  # Up
            else:
                return None  # Reached start, switch to forward
    else:  # Odd reverse rows go right
        if j < max_col:
            return [0, 1]  # Right
        else:
            if i > 0:
                return [-1, 0]  # Up
            else:
                return None  # Reached start, switch to forward


def lawnmower_vertical(current_pos):
    """
    Move the robot in a vertical lawnmower pattern.
    """
    max_row = len(world) - 1
    max_col = len(world[0]) - 1
    i, j = current_pos

    if j % 2 == 0:  # Even cols go down
        if i < max_row:
            return [1, 0]  # Down
        else:
            if j < max_col:
                return [0, 1]  # Right
            else:
                return None  # Reached end, switch to up
    else:  # Odd cols go up
        if i > 0:
            return [-1, 0]  # Up
        else:
            if j < max_col:
                return [0, 1]  # Right
            else:
                return None  # Reached end, switch to up


def lawnmower_vertical_reverse(current_pos):
    """
    Move the robot in a vertical lawnmower pattern in reverse.
    """
    max_row = len(world) - 1
    max_col = len(world[0]) - 1
    i, j = current_pos

    if (max_col - j) % 2 == 0:  # Even reverse cols go up
        if i > 0:
            return [-1, 0]  # Up
        else:
            if j > 0:
                return [0, -1]  # Left
            else:
                return None  # Reached start, switch to down
    else:  # Odd reverse cols go down
        if i < max_row:
            return [1, 0]  # Down
        else:
            if j > 0:
                return [0, -1]  # Left
            else:
                return None  # Reached start, switch to down


def manhattan_move(current_pos, goal_pos):
    """
    Move the robot towards the goal position using Manhattan distance.
    """
    dx = goal_pos[0] - current_pos[0]
    dy = goal_pos[1] - current_pos[1]

    if abs(dx) > abs(dy):
        return [np.sign(dx), 0]
    elif dy != 0:
        return [0, np.sign(dy)]
    else:
        return [0, 0]


def create_trajectory_animation(gt_trajectory, b_trajectory, world):
    """
    Create an animation of the robot's trajectory in the world.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create world background
    color_map = {'white': 0, 'yellow': 1, 'green': 2, 'meta': 3}
    world_numeric = [[color_map[cell] for cell in row] for row in world]
    cmap = ListedColormap(['floralwhite', 'khaki',
                           'mediumseagreen', 'fuchsia'])
    ax.imshow(world_numeric, cmap=cmap, vmin=0, vmax=3)

    # Set up grid and labels
    ax.set_xticks(np.arange(len(world[0])+1) - 0.5, [])
    ax.set_yticks(np.arange(len(world))+1 - 0.5, [])
    ax.grid(True, color='black', linewidth=1.5)

    # Add cell labels
    # for i in range(len(world)):
    #     for j in range(len(world[i])):
    #         ax.text(j, i, world[i][j], ha='center', va='center',
    #                 color='white' if world[i][j] == 'yellow' or
    #                 world[i][j] == 'green' else 'black')

    # Initialize trajectory lines
    gt_line, = ax.plot([], [], color='springgreen', linewidth=2,
                       alpha=0.9, label='Real Trajectory')
    b_line, = ax.plot([], [], color='blue',
                      linewidth=1, alpha=0.9, label='Belief Trajectory')
    robot_point = ax.scatter([0], [0], c='black',
                             s=100, label='Robot')  # Initial dummy pos

    # Initialize text annotation for step
    step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    ax.legend(loc='upper right')
    ax.set_title('Robot Localization Trajectories')

    def init():
        """
        Initialize the animation.
        """
        gt_line.set_data([], [])
        b_line.set_data([], [])
        robot_point.set_offsets([0, 0])
        step_text.set_text('')
        return gt_line, b_line, robot_point, step_text

    def update(frame):
        """
        Update the animation for each frame.
        """
        # Update ground truth trajectory
        x_gt = [pos[1] for pos in gt_trajectory[:frame+1]]
        y_gt = [pos[0] for pos in gt_trajectory[:frame+1]]
        gt_line.set_data(x_gt, y_gt)

        # Update belief trajectory
        if frame < len(b_trajectory):
            x_b = [pos[1] for pos in b_trajectory[:frame+1]]
            y_b = [pos[0] for pos in b_trajectory[:frame+1]]
            b_line.set_data(x_b, y_b)

        # Update robot position
        if frame < len(gt_trajectory):
            robot_point.set_offsets([gt_trajectory[frame][1],
                                     gt_trajectory[frame][0]])

        step_text.set_text(f'Step: {frame}')
        return gt_line, b_line, robot_point, step_text

    frames = max(len(gt_trajectory), len(b_trajectory))
    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                         interval=500, blit=True, repeat=False)

    plt.tight_layout()
    plt.show(block=False)
    return anim


def create_belief_animation(belief_states):
    """
    Create an animation of the belief distribution over time.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        """
        Update the belief distribution for each frame.
        """
        ax.clear()
        p = np.array(belief_states[frame])
        im = ax.imshow(p, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f"Belief Distribution - Step {frame}")
        ax.set_xticks(np.arange(len(p[0])+1) - 0.5)
        ax.set_yticks(np.arange(len(p)+1) - 0.5)
        ax.grid(True, color='black', linewidth=1.5)

        for i in range(len(p)):
            for j in range(len(p[i])):
                ax.text(j, i, f"{p[i][j]:.2f}", ha='center',
                        va='center', color='black')

        return [im]

    anim = FuncAnimation(fig, update, frames=len(belief_states),
                         interval=500, blit=False, repeat=False)
    plt.tight_layout()
    plt.show(block=False)
    return anim


world = [
    ['white', 'white', 'green', 'white', 'white', 'white', 'white',
     'white', 'white'],
    ['green', 'white', 'white', 'white', 'white', 'white', 'white',
     'yellow', 'white'],
    ['white', 'white', 'white', 'white', 'yellow', 'white', 'white',
     'white', 'white'],
    ['white', 'white', 'white', 'meta', 'white', 'white', 'white',
     'white', 'white'],
    ['yellow', 'white', 'white', 'white', 'white', 'white', 'white',
     'white', 'green']
]

# world = [
#     ['white', 'white', 'white', 'yellow', 'white', 'white', 'white',
# 'white', 'white'],
#     ['white', 'white', 'white', 'white', 'white', 'white', 'white',
# 'yellow', 'white'],
#     ['white', 'meta', 'white', 'white', 'yellow', 'white', 'green',
# 'white', 'white'],
#     ['white', 'white', 'green', 'white', 'white', 'white', 'white',
# 'white', 'white'],
#     ['white', 'white', 'white', 'white', 'white', 'white', 'white',
# 'white', 'green']
# ]

if __name__ == "__main__":
    sensor_right = 0.8  # Probability of correct measurement
    sensor_wrong = 1 - sensor_right  # Probability of wrong measurement

    p_move = 0.88  # Probability of moving in the intended direction
    p_stay = 1 - p_move  # Probability of staying in the same position

    # Initialize probabilities
    pinit = 1.0 / float(len(world)) / float(len(world[0]))
    p = [[pinit for row in range(len(world[0]))] for col in range(len(world))]

    gt_pos = [0, 0]  # Starting ground truth position
    gt_trajectory = [gt_pos.copy()]  # Ground truth trajectory
    b_pos, _ = get_most_probable_pos(p)
    b_trajectory = [b_pos.copy()]  # Belief trajectory
    belief_states = []  # Belief states for animation

    step = 0
    running = True

    goal_pos = find_goal(world)  # Find the goal position
    found = False  # Flag to indicate if the goal has been found
    max_steps = 70  # Maximum number of steps to run
    current_lawnmower = lawnmower_vertical

    while running and step < max_steps:
        if found is False:
            measurement = get_measurement(gt_pos[0], gt_pos[1])
            if measurement == 'meta':
                measurement = 'yellow'

            # chose motion to horizontal or vertical
            # motion = lawnmower_horizontal(gt_pos)
            motion = current_lawnmower(gt_pos)
            if motion is None:
                if current_lawnmower == lawnmower_vertical:
                    current_lawnmower = lawnmower_vertical_reverse
                else:
                    current_lawnmower = lawnmower_vertical

                motion = current_lawnmower(gt_pos)

            # update belief
            p = sense(p, world, measurement,
                      sensor_right=sensor_right, sensor_wrong=sensor_wrong)
            p = move(p, motion, p_move=p_move, p_stay=p_stay)

            # update real position (ground truth)
            gt_pos = update_real_postion(gt_pos, motion)
            gt_trajectory.append(gt_pos.copy())

            # most probable position (belief)
            b_pos, b_prob = get_most_probable_pos(p)
            b_trajectory.append(b_pos.copy())
            belief_states.append([row.copy() for row in p])

            print(f"\nStep {step}:")
            print(f"Ground truth position: {gt_pos}")
            print(f"Most probable position: {b_pos} , prob: {b_prob}")
            print(f"Goal position: {goal_pos}")
            print("Probability distribution:")
            show(p)

            if b_prob > 0.6:
                found = True
                print("\n¡Location estimated with sufficient certainty!")
                print("yellowirecting to the goal...")
        else:
            if gt_pos == goal_pos:
                print("\n¡The robot has reached the finish line!")
                running = False
            else:
                motion = manhattan_move(gt_pos, goal_pos)
                # update real position (ground truth)
                gt_pos = update_real_postion(gt_pos, motion)
                gt_trajectory.append(gt_pos.copy())
        step += 1

    print("\nThreshold reached! Showing trajectory animation...")

    print(f"size gt_trajectory: {len(gt_trajectory)}")
    print(f"size b_trajectory: {len(b_trajectory)}")

    # Create and show the animation
    anim = create_trajectory_animation(gt_trajectory, b_trajectory, world)
    anim_belief = create_belief_animation(belief_states)
    plt.pause(0.01)
    input("Press enter to continue...")
