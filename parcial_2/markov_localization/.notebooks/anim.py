import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import matplotlib
# Increase animation size limit
matplotlib.rcParams['animation.embed_limit'] = 100

# Load the data
# Adjust filename as needed
data = pd.read_csv('../.data/log_2020_04_11_18_04_22_266826.csv')
data.columns = data.columns.str.strip()
# Define the world map (should match your physical setup)
world = [
    ['white', 'white', 'green', 'white', 'white', 'white', 'white',
     'white', 'white'],
    ['green', 'white', 'white', 'white', 'white', 'white', 'white',
     'yellow', 'white'],
    ['white', 'white', 'white', 'white', 'yellow', 'white', 'white',
     'white', 'white'],
    ['white', 'white', 'white', 'red', 'white', 'white', 'white',
     'white', 'white'],
    ['yellow', 'white', 'white', 'white', 'white', 'white', 'white',
     'white', 'green']
]

# Create numeric representation for visualization
color_map = {'white': 0, 'yellow': 1, 'green': 2,
             'red': 3, 'meta': 3, 'None': 0}
world_numeric = np.array([[color_map[cell] for cell in row] for row in world])

# Prepare the trajectories
gt_trajectory = list(zip(data['gt_x'], data['gt_y']))
b_trajectory = list(zip(data['b_x'], data['b_y']))

# Create the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: World with Trajectories
cmap = ListedColormap(['floralwhite', 'khaki', 'mediumseagreen', 'fuchsia'])
ax1.imshow(world_numeric, cmap=cmap, vmin=0, vmax=3)

# Set up grid
for i in range(world_numeric.shape[0]+1):
    ax1.axhline(i-0.5, color='black', linewidth=1)
for j in range(world_numeric.shape[1]+1):
    ax1.axvline(j-0.5, color='black', linewidth=1)

# Add cell labels
# for i in range(len(world)):
#     for j in range(len(world[i])):
#         ax1.text(j, i, world[i][j], ha='center', va='center',
#                 color='white' if world[i][j] in ['yellow', 'green',
#                       'red'] else 'black')

# Initialize trajectory lines
gt_line, = ax1.plot([], [], 'go-', linewidth=2,
                    markersize=8, label='Real Path')
b_line, = ax1.plot([], [], 'bo--', linewidth=2,
                   markersize=8, label='Belief Path')
current_gt = ax1.plot([], [], 'ro',
                      markersize=12, label='Current Position')[0]
current_b = ax1.plot([], [], 'mo',
                     markersize=12, label='Belief Position')[0]

ax1.set_title('Robot Localization Trajectories')
ax1.legend(loc='upper right')
ax1.set_xlim(-0.5, len(world[0])-0.5)
ax1.set_ylim(len(world)-0.5, -0.5)  # Invert y-axis to match matrix coordinates

# Plot 2: Belief Probability Over Time
ax2.set_xlim(0, len(data))
ax2.set_ylim(0, 1)
ax2.set_xlabel('Step')
ax2.set_ylabel('Probability')
ax2.set_title('Localization Confidence Over Time')
ax2.axhline(0.6, color='red', linestyle='--', label='Threshold')
prob_line, = ax2.plot([], [], 'b-', label='Belief Probability')
ax2.legend()

# Add step annotation
step_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
measure_text = ax1.text(0.02, 0.90, '', transform=ax1.transAxes, color='red')


def init():
    gt_line.set_data([], [])
    b_line.set_data([], [])
    current_gt.set_data([], [])
    current_b.set_data([], [])
    prob_line.set_data([], [])
    step_text.set_text('')
    measure_text.set_text('')
    return (gt_line, b_line, current_gt, current_b,
            prob_line, step_text, measure_text)


def update(frame):
    # Update trajectories
    x_gt = [p[1] for p in gt_trajectory[:frame+1]]
    y_gt = [p[0] for p in gt_trajectory[:frame+1]]
    gt_line.set_data(x_gt, y_gt)

    x_b = [p[1] for p in b_trajectory[:frame+1]]
    y_b = [p[0] for p in b_trajectory[:frame+1]]
    b_line.set_data(x_b, y_b)

    # Update current positions
    current_gt.set_data([gt_trajectory[frame][1]], [gt_trajectory[frame][0]])
    current_b.set_data([b_trajectory[frame][1]], [b_trajectory[frame][0]])

    # Update probability plot
    prob_line.set_data(data['step'][:frame+1], data['b_prob'][:frame+1])

    # Update text annotations
    step_text.set_text(f'Step: {frame}')
    measure_text.set_text(f'Measurement: {data["measurement"][frame]}')

    return (gt_line, b_line, current_gt, current_b,
            prob_line, step_text, measure_text)


# Create animation
ani = FuncAnimation(fig, update, frames=len(data), init_func=init,
                    interval=500, blit=True)

# Display the animation
plt.show(block=False)  # Show the plot without blocking
input("Press Enter to close the plot...")  # Wait for user input
plt.close()  # Prevents duplicate display
# HTML(ani.to_jshtml())
