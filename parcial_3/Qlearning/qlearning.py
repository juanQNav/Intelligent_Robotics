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

actions = [0, 1, 2, 3]  # Up, Down, Left, Right

Q = np.zeros((45, 4))  # 45 states, 4 possible actions per state


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
        return valid_actions

    def move(self, state, action):
        i, j = self.state_to_location[state]
        if action == 0:
            i -= 1
        elif action == 1:
            i += 1
        elif action == 2:
            j -= 1
        elif action == 3:
            j += 1
        return self.location_to_state[(i, j)]

    def training(self, start_location, end_location, iterations):
        end_state = self.location_to_state[end_location]
        for _ in range(iterations):
            current_state = np.random.randint(0, len(self.state_to_location))
            valid_actions = self.get_valid_actions(current_state)
            if not valid_actions:
                continue
            action = np.random.choice(valid_actions)
            next_state = self.move(current_state, action)
            reward = 100 if next_state == end_state else -1
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


# ==== Visualization ====
def location_to_coords(location):
    return location  # location is already (i, j)


def create_trajectory_animation(gt_trajectory, world):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = ListedColormap(['floralwhite', 'khaki',
                           'mediumseagreen', 'fuchsia'])
    ax.imshow(world, cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(np.arange(len(world[0])+1) - 0.5, [])
    ax.set_yticks(np.arange(len(world))+1 - 0.5, [])
    ax.grid(True, color='black', linewidth=1.5)

    gt_line, = ax.plot([], [], color='springgreen',
                       linewidth=2, alpha=0.9, label='Ruta')
    robot_point = ax.scatter([0], [0], c='black', s=100, label='Robot')
    step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.legend(loc='upper right')
    ax.set_title('Trayectoria del Robot')

    def init():
        gt_line.set_data([], [])
        robot_point.set_offsets([0, 0])
        step_text.set_text('')
        return gt_line, robot_point, step_text

    def update(frame):
        x_gt = [pos[1] for pos in gt_trajectory[:frame+1]]
        y_gt = [pos[0] for pos in gt_trajectory[:frame+1]]
        gt_line.set_data(x_gt, y_gt)
        if frame < len(gt_trajectory):
            robot_point.set_offsets([x_gt[-1], y_gt[-1]])
        step_text.set_text(f'Paso: {frame}')
        return gt_line, robot_point, step_text

    anim = FuncAnimation(fig, update,
                         frames=len(gt_trajectory), init_func=init,
                         interval=500, blit=True, repeat=False)
    plt.tight_layout()
    plt.show(block=False)
    return anim


# ==== Main ====
if __name__ == "__main__":
    agent = QAgent(alpha, gamma, location_to_state,
                   actions, world, state_to_location, Q)

    start = (4, 0)  # L9
    end = (3, 3)    # L1

    agent.training(start, end, 1000)

    print("Route found:", agent.optimal_route)

    gt_trajectory = [location_to_coords(loc) for loc in agent.optimal_route]
    anim = create_trajectory_animation(gt_trajectory, world)
    plt.pause(0.01)
    plt.show(block=False)
    input("Press Enter to close the animation...")
    plt.close()

    # Save the Q-table to a CSV file
    df_q = pd.DataFrame(agent.Q, columns=['Up', 'Down', 'Left', 'Right'])
    df_q.index.name = 'State'
    df_q.to_csv('q_table.csv', index=True)
    print("Q-table saved to 'q_table.csv'")

    # show the Q-table
    print("Q-table:")
    print(df_q)
