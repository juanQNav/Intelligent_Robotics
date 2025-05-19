from matplotlib.animation import FuncAnimation, PillowWriter
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


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


def create_environment(size, noise_level=0.1, num_shades=5):
    env = np.zeros((size, size), dtype=float)
    center = size // 2
    max_concentration = 1.0

    # Create a gradient of concentration from the center to the borders
    for i in range(size):
        for j in range(size):
            # Chebyshev distance
            distance = max(abs(i - center), abs(j - center))
            env[i, j] = max(0, max_concentration - distance / center)

    # Discretize the environment into num_shades levels
    levels = np.linspace(0, 1, num_shades)  # Equally spaced levels
    # Assign a shade to each cell
    env_discretized = np.digitize(env, levels) - 1
    env_discretized = env_discretized / (num_shades - 1)  # Normalize to [0, 1]

    # Add Gaussian noise to the environment
    noise = np.random.normal(loc=0, scale=noise_level, size=(size, size))
    env_with_noise = env_discretized + noise

    # Ensure the values are in the [0, 1] range
    env_with_noise = np.clip(env_with_noise, 0, 1)

    return env_with_noise


def run(env_size, x, y, angle):
    dx = int(np.round(np.cos(angle)))
    dy = int(np.round(np.sin(angle)))
    new_x = max(0, min(env_size - 1, x + dx))
    new_y = max(0, min(env_size - 1, y + dy))

    return new_x, new_y


def tumble(angle):
    return random.uniform(-np.pi/2, np.pi/2) + angle  # Random new direction


def sensing_color_reflection(env, x, y):
    return env[x, y]


def run_simulation(env_size, a, b, threshold, noise_level=0.1,
                   num_shades=5, steps=1000, max_run=3, base_run=1):
    env = create_environment(env_size, noise_level, num_shades)

    # Select a random edge (0: up, 1: down, 2: left, 3: right)
    edge = random.choice([0, 1, 2, 3])

    # Initialize the bacterium at the selected edge
    if edge == 0:  # Up edge
        position = [0, random.randint(0, env_size - 1)]
    elif edge == 1:  # Down edge
        position = [env_size - 1, random.randint(0, env_size - 1)]
    elif edge == 2:  # Left edge
        position = [random.randint(0, env_size - 1), 0]
    elif edge == 3:  # Right edge
        position = [random.randint(0, env_size - 1), env_size - 1]

    leaky_integrator = LeakyIntegrator(
        a, b, threshold, sensing_color_reflection(env, position[0],
                                                  position[1]))

    angle = random.uniform(0, 2 * np.pi)  # Start with a random direction

    path = [tuple(position)]
    path_X = [sensing_color_reflection(env, position[0], position[1])]
    path_V = [0]
    path_U = [0]

    for step in range(steps):
        x, y = position
        X = sensing_color_reflection(env, x, y)
        V = leaky_integrator.update(X)
        U = X - V
        behavior = leaky_integrator.get_behavior(X)

        if behavior == "run":
            for _ in range(max_run):
                x, y = position
                new_x, new_y = run(env_size, x, y, angle)
                position = [new_x, new_y]
                path.append(tuple(position))
                path_X.append(X)
                path_V.append(V)
                path_U.append(U)
        else:
            angle = tumble(angle)
            for _ in range(base_run):
                x, y = position
                new_x, new_y = run(env_size, x, y, angle)
                position = [new_x, new_y]
                path.append(tuple(position))
                path_X.append(X)
                path_V.append(V)
                path_U.append(U)

        print(
            f"Step: {step}, X: {X:.2f}, V:{V:.2f}  U(t): {U},\
             Behavior: {behavior}")

    return path, env, path_X, path_V, path_U


def update_combined_plot(frame, path, scat, trajectory, line_x, line_v, line_u,
                         X_values, V_values, U_values, threshold_line):
    # Update simulation plot
    x, y = path[frame]
    scat.set_offsets([y, x])
    trajectory.set_data([p[1] for p in path[:frame+1]],
                        [p[0] for p in path[:frame+1]])

    # Update time series plot
    line_x.set_data(range(frame+1), X_values[:frame+1])
    line_v.set_data(range(frame+1), V_values[:frame+1])
    line_u.set_data(range(frame+1), U_values[:frame+1])

    return scat, trajectory, line_x, line_v, line_u


if __name__ == '__main__':
    env_size = 150
    a = 0.5  # Decay rate
    b = 0.5  # Sensitivity to changes
    threshold = 0.025  # Threshold for behavior change
    noise_level = 0.02
    num_shades = 10  # Number of shades in the environment
    max_run = 5
    base_run = 1
    steps = 500
    save_gif = True

    print("--"*50)
    print("E. coli Simulation (Run & Tumble)")

    path, env, X_values, V_values, U_values = run_simulation(
        env_size, a, b, threshold, noise_level, num_shades,
        steps, max_run, base_run)

    # Create a combined figure
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    # Simulation plot
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(env, cmap='Greens', interpolation='nearest')
    ax1.set_title('Simulation E. coli (Run & Tumble)')
    scat = ax1.scatter([], [], color='red', s=50)
    trajectory, = ax1.plot([], [], color='blue', lw=1)

    # Time series plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('X, V, and U(t) over time')
    line_x, = ax2.plot([], [], label='X')
    line_v, = ax2.plot([], [], label='V')
    line_u, = ax2.plot([], [], label='U(t)')
    threshold_line = ax2.axhline(y=threshold, color='r', linestyle='--',
                                 label='Threshold')
    ax2.legend()
    ax2.set_xlim(0, len(X_values))
    ax2.set_ylim(min(min(X_values), min(V_values), min(U_values))-0.05,
                 max(max(X_values), max(V_values), max(U_values))+0.05)

    # Adjust layout
    plt.tight_layout()

    # Create animation
    frame_skip = 3
    ani = FuncAnimation(fig,
                        update_combined_plot,
                        frames=range(
                            0, len(path), frame_skip), fargs=(
                                path, scat, trajectory, line_x, line_v, line_u,
                                X_values, V_values, U_values,
                                threshold_line), interval=100, blit=True)

    # Save as GIF
    if save_gif:
        print("Saving animation...")
        writer = PillowWriter(fps=15)
        ani.save("combined_animation.gif", writer=writer)
        print("Animation saved as combined_animation.gif")

    plt.show(block=False)

    print("--"*50)
    input("Press any key to exit...")
