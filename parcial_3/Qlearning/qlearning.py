import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import pandas as pd

# ==== World map ====
world = [
    [0, 0, 2, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 2]
]

# ==== Q-learning setup ====
gamma = 0.75
alpha = 0.9

location_to_state = {}
state_to_location = {}
state = 0
for i in range(len(world)):
    for j in range(len(world[0])):
        location_to_state[(i, j)] = state
        state_to_location[state] = (i, j)
        state += 1

actions = [0, 1, 2, 3, 4]  # Up, Down, Left, Right, Stop

Q = np.zeros((45, 5))  # 45 states, 5 possible actions per state

rewards = np.full((45, 5), -1)  # Initialize rewards with -1


class QAgent:
    def __init__(self, alpha, gamma, location_to_state, actions, rewards,
                 state_to_location, Q):
        self.gamma = gamma
        self.alpha = alpha
        self.location_to_state = location_to_state
        self.actions = actions
        self.rewards = rewards
        self.state_to_location = state_to_location
        self.Q = Q
        self.optimal_route = []

    def get_valid_actions(self, state):
        i, j = self.state_to_location[state]
        valid_actions = []
        if i > 0:
            valid_actions.append(0)  # Up
        if i < len(world) - 1:
            valid_actions.append(1)  # Down
        if j > 0:
            valid_actions.append(2)  # Left
        if j < len(world[0]) - 1:
            valid_actions.append(3)  # Right
        if world[i][j] == 3:
            valid_actions.append(4)  # Stop if at goal
        return valid_actions

    def move(self, state, action):
        i, j = self.state_to_location[state]
        if action == 0 and i > 0:
            i -= 1
        elif action == 1 and i < len(world) - 1:
            i += 1
        elif action == 2 and j > 0:
            j -= 1
        elif action == 3 and j < len(world[0]) - 1:
            j += 1
        elif action == 4:
            return state
        else:
            return state
        return self.location_to_state[(i, j)]

    def training(self, start_location, end_location, iterations):
        end_state = self.location_to_state[end_location]
        rewards[end_state, 4] = 999  # Reward for reaching the goal
        for _ in range(iterations):
            current_state = np.random.randint(0, len(self.state_to_location))
            valid_actions = self.get_valid_actions(current_state)
            if not valid_actions:
                continue
            action = np.random.choice(valid_actions)
            next_state = self.move(current_state, action)
            reward = self.rewards[current_state, action]
            TD = reward + self.gamma * np.max(self.Q[next_state]) - \
                self.Q[current_state, action]
            self.Q[current_state, action] += self.alpha * TD

        # Compute optimal route
        route = [start_location]
        next_location = start_location
        self.get_optimal_route(start_location, end_location,
                               next_location, route, self.Q)
        self.optimal_route = route

    def get_optimal_route(self, start_location, end_location,
                          next_location, route, Q):
        max_steps = 100
        steps = 0
        while next_location != end_location and steps < max_steps:
            state = self.location_to_state[start_location]
            action = np.argmax(Q[state])
            next_state = self.move(state, action)
            next_location = self.state_to_location[next_state]
            route.append(next_location)
            start_location = next_location
            steps += 1


# ==== Markov's localization ====
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
            if world[i][j] == 3:
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


def update_postion(gt_pos, motion):
    """
    Update the ground truth position based on the motion vector.
    """
    gt_pos[0] += motion[0]
    gt_pos[1] += motion[1]
    gt_pos[0] = max(0, min(gt_pos[0], len(world)-1))
    gt_pos[1] = max(0, min(gt_pos[1], len(world[0])-1))

    return gt_pos


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


# ==== Visualization ====
def location_to_coords(location):
    return location  # location is already (i, j)


# ==== Trajectory Animation ====
def create_trajectory_animation(gt_trajectory, b_trajectory, world):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = ListedColormap(['floralwhite', 'khaki',
                           'mediumseagreen', 'fuchsia'])
    ax.imshow(world, cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(np.arange(len(world[0])+1) - 0.5, [])
    ax.set_yticks(np.arange(len(world)+1) - 0.5, [])
    ax.grid(True, color='black', linewidth=1.5)

    gt_line, = ax.plot([], [], color='springgreen',
                       linewidth=2, alpha=0.9, label='Real Trajectory')
    if b_trajectory is not None:
        b_line, = ax.plot([], [], color='blue',
                          linewidth=1, alpha=0.9, label='Belief Trajectory')
    robot_point = ax.scatter([0], [0], c='black', s=100, label='Robot')
    step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.legend(loc='upper right')
    ax.set_title('Trayectoria del Robot')

    def init():
        gt_line.set_data([], [])
        if b_trajectory is not None:
            b_line.set_data([], [])
        robot_point.set_offsets([0, 0])
        step_text.set_text('')
        if b_trajectory is None:
            return gt_line, robot_point, step_text
        else:
            return gt_line, b_line, robot_point, step_text

    def update(frame):
        x_gt = [pos[1] for pos in gt_trajectory[:frame+1]]
        y_gt = [pos[0] for pos in gt_trajectory[:frame+1]]
        gt_line.set_data(x_gt, y_gt)

        if b_trajectory is not None:
            if frame < len(b_trajectory):
                x_b = [pos[1] for pos in b_trajectory[:frame+1]]
                y_b = [pos[0] for pos in b_trajectory[:frame+1]]
                b_line.set_data(x_b, y_b)

        if frame < len(gt_trajectory):
            robot_point.set_offsets([x_gt[-1], y_gt[-1]])
        step_text.set_text(f'Paso: {frame}')

        if b_trajectory is None:
            return gt_line, robot_point, step_text
        else:
            return gt_line, b_line, robot_point, step_text

    if b_trajectory is not None:
        frames = max(len(gt_trajectory), len(b_trajectory))
    else:
        frames = len(gt_trajectory)
    anim = FuncAnimation(fig, update,
                         frames=frames, init_func=init,
                         interval=500, blit=True, repeat=False)
    plt.tight_layout()
    plt.show(block=False)
    return anim


# ==== Belfief Animation ====
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


# ==== Main ====
if __name__ == "__main__":
    # ==== Q-learning ====
    agent = QAgent(alpha, gamma, location_to_state,
                   actions, rewards, state_to_location, Q)

    start = (3, 8)
    end = (3, 3)

    agent.training(start, end, 1000)

    print("Route found:", agent.optimal_route)

    gt_trajectory = [location_to_coords(loc) for loc in agent.optimal_route]
    anim = create_trajectory_animation(gt_trajectory=gt_trajectory,
                                       b_trajectory=None,
                                       world=world)
    plt.pause(0.01)
    plt.show(block=False)
    input("Press Enter to close the animation...")
    plt.close()

    # Save the Q-table to a CSV file
    df_q = pd.DataFrame(agent.Q, columns=['Up', 'Down', 'Left',
                                          'Right', 'Stop'])
    df_q.index.name = 'State'
    df_q.to_csv('q_table.csv', index=True)
    print("Q-table saved to 'q_table.csv'")

    # show the Q-table
    print("Q-table:")
    print(df_q)

    # ==== Markov's localization ====
    sensor_right = 0.8  # Probability of correct measurement
    sensor_wrong = 1 - sensor_right  # Probability of wrong measurement

    p_move = 0.88  # Probability of moving in the intended direction
    p_stay = 1 - p_move  # Probability of staying in the same position

    # Initialize probabilities
    pinit = 1.0 / float(len(world)) / float(len(world[0]))
    p = [[pinit for row in range(len(world[0]))] for col in range(len(world))]

    gt_pos = [4, 2]  # Starting ground truth position
    gt_trajectory = [gt_pos.copy()]  # Ground truth trajectory
    b_pos, _ = get_most_probable_pos(p)
    b_trajectory = [b_pos.copy()]  # Belief trajectory
    belief_states = []  # Belief states for animation

    step = 0
    running = True

    goal_pos = find_goal(world)  # Find the goal position
    localized = False  # Flag to check if the robot is localized
    max_steps = 70  # Maximum number of steps to run
    current_lawnmower = lawnmower_vertical

    while running and step < max_steps:
        if localized is False:
            measurement = get_measurement(gt_pos[0], gt_pos[1])
            if measurement == 'meta':
                measurement = 'yellow'

            # chose motion to horizontal or vertical
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
            gt_pos = update_postion(gt_pos, motion)
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
                localized = True
                print("\n¡Location estimated with sufficient certainty!")
                print("yellowirecting to the goal...")
        else:
            # Use the learned Q-policy to go to the goal
            while gt_pos != goal_pos and step < max_steps:
                state = location_to_state[tuple(gt_pos)]
                action = np.argmax(agent.Q[state])
                next_state = agent.move(state, action)
                gt_pos = list(state_to_location[next_state])
                gt_trajectory.append(gt_pos.copy())
                b_trajectory.append(gt_pos.copy())
                belief_states.append([row.copy() for row in p])

                step += 1
            print("\n¡The robot has reached the finish line!")
            running = False
        step += 1

    print("\nThreshold reached! Showing trajectory animation...")

    print(f"size gt_trajectory: {len(gt_trajectory)}")
    print(f"size b_trajectory: {len(b_trajectory)}")

    # Create and show the animation
    anim = create_trajectory_animation(gt_trajectory, b_trajectory, world)
    anim_belief = create_belief_animation(belief_states)
    plt.pause(0.01)
    input("Press enter to continue...")
