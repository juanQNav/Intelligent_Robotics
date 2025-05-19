# Juan Luis Quistian Navarro
"""
    Simulate a markov's localization to gloabal localization problem.
    The robot is in a grid world and has to localize itself.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import csv
from datetime import datetime
import os
from typing import List, Tuple, Optional, Any


def sense(p: List[List[float]], world: List[List[str]], measurement: str,
          sensor_right: float, sensor_wrong: float) -> List[List[float]]:
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


def move(p: List[List[float]], motion: List[int],
         p_move: float, p_stay: float) -> List[List[float]]:
    """
    Move the robot in the grid with a given motion vector.
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


def show(p: List[List[float]]) -> None:
    """
    Show the probability distribution in a grid format.
    """
    for i in range(len(p)):
        for j in range(len(p[i])):
            print(f"{p[i][j]:.2f} |", end=" ")
        print("\n"+"-"*62)


def find_goal(world: List[List[str]]) -> Optional[List[int]]:
    """
    Find the position of the goal in the world.
    """
    for i in range(len(world)):
        for j in range(len(world[0])):
            if world[i][j] == 'meta':
                return [i, j]
    return None


def get_measurement(pos_x: int, pos_y: int) -> str:
    """
    Get the measurement from the world based on the robot's position.
    """

    return world[pos_x][pos_y]


def get_most_probable_pos(p: List[List[float]]) -> Tuple[List[int], float]:
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


def update_real_postion(gt_pos: List[int], motion: List[int]) -> List[int]:
    """
    Update the ground truth position based on the motion vector.
    """
    gt_pos[0] += motion[0]
    gt_pos[1] += motion[1]
    gt_pos[0] = max(0, min(gt_pos[0], len(world)-1))
    gt_pos[1] = max(0, min(gt_pos[1], len(world[0])-1))

    return gt_pos


def lawnmower_horizontal(current_pos: List[int]) -> List[int]:
    """
    Move the robot in a horizontal lawnmower pattern.
    """
    max_row = len(world)
    max_col = len(world[0])

    i, j = current_pos

    # Determine if we are in a left-to-right row (even row index)
    if i % 2 == 0:
        if j < max_col - 1:
            return [0, 1]  # move right
        elif i < max_row - 1:
            return [1, 0]  # move down to next row
        else:
            return [0, 0]  # end of mape
    else:
        if j > 0:
            return [0, -1]  # move left
        elif i < max_row - 1:
            return [1, 0]  # move down o next row
        else:
            return [0, 0]  # end of map


def lawnmower_vertical(current_pos: List[int]) -> Optional[List[int]]:
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


def lawnmower_vertical_reverse(current_pos: List[int]) -> Optional[List[int]]:
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


def manhattan_move(current_pos: List[int], goal_pos: List[int]) -> List[int]:
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


def create_trajectory_animation(gt_trajectory: List[List[int]],
                                b_trajectory: List[List[int]],
                                world: List[List[str]],
                                config_name: str,
                                pos_idx: int,
                                strategy: str) -> Any:
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
    # ax.set_title('Robot Localization Trajectories')

    def init():
        gt_line.set_data([], [])
        b_line.set_data([], [])
        robot_point.set_offsets([0, 0])
        step_text.set_text('')
        return gt_line, b_line, robot_point, step_text

    def update(frame):
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

    return anim, fig, update, frames


def create_belief_animation(belief_states: List[List[List[float]]]) -> Any:
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
    # plt.show(block=False)
    return anim


def plot_localization_confidence(belief_states: List[List[List[float]]],
                                 config_name: str,
                                 pos_idx: int,
                                 strategy: str,
                                 threshold: float = 0.6) -> None:
    """
    Plot the localization confidence over time and save as an image.
    """
    # Extract max probability at each step
    max_probs = [max(max(row) for row in state) for state in belief_states]
    steps = range(len(max_probs))

    plt.figure(figsize=(10, 6))

    # Plot confidence over time
    plt.plot(steps, max_probs, 'b-',
             linewidth=2, label='Localization Confidence'
             )

    # Plot threshold line
    plt.axhline(y=threshold, color='r', linestyle='--',
                label=f'Threshold ({threshold})')

    # Find and mark when threshold was crossed
    above_threshold = [i for i,
                       prob in enumerate(max_probs) if prob >= threshold]
    if above_threshold:
        first_above = above_threshold[0]
        plt.scatter(first_above, max_probs[first_above], color='g', s=100,
                    label=f'Localized (step {first_above})')

    plt.xlabel('Time Step')
    plt.ylabel('Maximum Probability')
    # plt.title('Localization Confidence Over Time')
    plt.legend()
    plt.grid(True)

    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig(
        f"results/{config_name}_pos{pos_idx}_{strategy}_confidence.png"
    )
    plt.close()


def run_simulation(world: List[List[str]],
                   gt_pos: List[int],
                   strategy: str,
                   pos_idx: int,
                   config_name: str,
                   show_anim: bool = False) -> None:
    """
    Run the simulation for a given world configuration and initial position.
    """

    # Initialize CSV file and write header
    os.makedirs('results', exist_ok=True)
    csv_filename = "results/simulation_results.csv"
    header = [
        "timestamp", "config_name", "initial_pos", "strategy",
        "localization_steps", "localization_prob",
        "manhattan_distance", "steps_to_goal", "success"
    ]

    # Only write header if file doesn't exist
    try:
        with open(csv_filename, 'x', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
    except FileExistsError:
        pass

    # Initialize probabilities
    pinit = 1.0 / float(len(world)) / float(len(world[0]))
    p = [[pinit for row in range(len(world[0]))] for col in range(len(world))]

    gt_trajectory = [gt_pos.copy()]  # Ground truth trajectory
    b_pos, _ = get_most_probable_pos(p)
    b_trajectory = [b_pos.copy()]  # Belief trajectory
    belief_states = []  # Belief states for animation

    step = 0
    running = True
    localization_steps = 0
    localization_prob = 0.0
    localized = False
    goal_reached = False

    goal_pos = find_goal(world)  # Find the goal position
    max_steps = 100
    current_lawnmower = lawnmower_vertical

    while running and step <= max_steps:
        if not localized:
            measurement = get_measurement(gt_pos[0], gt_pos[1])
            if measurement == 'meta':
                measurement = 'yellow'

            # choose motion to horizontal or vertical
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
            if motion is not None:
                p = move(p, motion, p_move=p_move, p_stay=p_stay)

            # update real position (ground truth)
            if motion is not None:
                gt_pos = update_real_postion(gt_pos, motion)
            else:
                print("Motion is None, skipping position update.")
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

            if b_prob > 0.6 and not localized:
                localized = True
                localization_steps = step
                localization_prob = b_prob
                print("\n¡Location estimated with sufficient certainty!")
                print("Redirecting to the goal...")
        else:
            if gt_pos == goal_pos:
                print("\n¡The robot has reached the finish line!")
                goal_reached = True
                running = False
            else:
                if goal_pos is not None:
                    motion = manhattan_move(gt_pos, goal_pos)
                else:
                    print("Error: Goal position is None. Exiting simulation.")
                    break
                # update real position (ground truth)
                gt_pos = update_real_postion(gt_pos, motion)
                gt_trajectory.append(gt_pos.copy())
        step += 1

    # Calculate metrics for CSV
    if localized:
        # Manhattan distance at localization
        if goal_pos is not None:
            manhattan_dist = abs(
                b_pos[0] - goal_pos[0]) + abs(b_pos[1] - goal_pos[1])
        else:
            manhattan_dist = -1  # Set to -1 if goal_pos is None
        # Steps to goal after localization (only if goal was reached)
        steps_after_localization = (
            step - localization_steps if goal_reached else -1
        )
    else:
        manhattan_dist = -1
        steps_after_localization = -1

    # Write results to CSV
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config_name,
            f"({initial_pos[0]},{initial_pos[1]})",
            strategy,
            localization_steps if localized else -1,
            localization_prob if localized else -1,
            manhattan_dist,
            steps_after_localization,
            goal_reached
        ])

    print("\nThreshold reached! Showing trajectory animation...")
    print(f"size gt_trajectory: {len(gt_trajectory)}")
    print(f"size b_trajectory: {len(b_trajectory)}")

    # Plot localization confidence
    plot_localization_confidence(belief_states, config_name, pos_idx, strategy)

    # Create and show the animation
    anim, fig, update, frames = create_trajectory_animation(
        gt_trajectory,
        b_trajectory, world,
        config_name,
        pos_idx,
        strategy
    )
    if show_anim:
        plt.pause(0.01)
        plt.show(block=False)
        # Save the animation
    anim.save(f"results/{config_name}_pos{pos_idx}_{strategy}.gif",
              writer='imagemagick')

    # Save the final frame as an image
    # Update to the last frame
    update(frames-1)
    plt.savefig(f"results/{config_name}_pos{pos_idx}_{strategy}_final.png")

    anim_belief = create_belief_animation(belief_states)
    if show_anim:
        plt.pause(0.01)
        plt.show(block=False)
    anim_belief.save(
        f"results/{config_name}_belief_pos{pos_idx}_{strategy}.gif",
        writer='imagemagick'
    )
    if show_anim:
        input("Press enter to continue...")
    plt.close('all')


world_configurations = {
    'config1': [
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
    ],
    'config2': [
        ['white', 'white', 'white', 'yellow', 'white', 'white', 'white',
         'white', 'white'],
        ['white', 'white', 'white', 'white', 'white', 'white', 'white',
         'yellow', 'white'],
        ['white', 'meta', 'white', 'white', 'yellow', 'white', 'green',
         'white', 'white'],
        ['white', 'white', 'green', 'white', 'white', 'white', 'white',
         'white', 'white'],
        ['white', 'white', 'white', 'white', 'white', 'white', 'white',
         'white', 'green']
    ],
    'config3': [
        ['white', 'white', 'white', 'white', 'white', 'white', 'white',
         'white', 'green'],
        ['white', 'yellow', 'white', 'white', 'meta', 'white', 'white',
         'white', 'white'],
        ['white', 'white', 'white', 'white', 'white', 'white', 'green',
         'white', 'white'],
        ['white', 'white', 'yellow', 'white', 'white', 'white', 'white',
         'white', 'white'],
        ['white', 'white', 'white', 'white', 'yellow', 'white', 'white',
         'green', 'white']
    ],
}

initial_positions = [
    [0, 0],  # 0
    [0, 8],  # 1
    [4, 0],  # 2
    [4, 8],  # 3
    [2, 4]   # 4
]

if __name__ == "__main__":
    sensor_right = 0.8  # Probability of correct measurement
    sensor_wrong = 1 - sensor_right  # Probability of wrong measurement

    p_move = 0.88  # Probability of moving in the intended direction
    p_stay = 1 - p_move  # Probability of staying in the same position

    for config_name, world in world_configurations.items():
        for pos_idx, initial_pos in enumerate(initial_positions):
            # for strategy in ['vertical']:
            run_simulation(world,
                           gt_pos=initial_pos.copy(),
                           strategy='vertical',
                           pos_idx=pos_idx,
                           config_name=config_name,
                           show_anim=False)  # change to False to not show anim
