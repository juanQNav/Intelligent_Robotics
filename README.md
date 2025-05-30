# Intelligent Robotics

This repository contains the work for the Intelligent Robotics course in the ISI program at UASLP.

# Content
The repository is organized by partials and includes:

In each partial we work with micropython to control Lego Ev3 and we use python to make simulations.

### **Partial 1**
- **Interaction With Environment**
- **Robotic Architectures**

We simulate the chemotactic behavior of E. coli, using a **leaky integrator** to make a reactive robot.

The simulation environment is created using iris sheets in five shades of green, representing different attractant concentrations. Attractant concentration values are captured using the robot's color sensor.

Each time the robot measures the attractant concentration, it updates its internal state \( V(t) \) using the differential equation:

$$
    \frac{dV(t)}{dt} + aV(t) = bX(t)
$$

Then, the robot's behavior is determined by comparing the difference \( U(t) = X(t) - V(t) \) with a predefined threshold.

<div align="center">
  <img src="multimedia/animation_ecoli.gif" alt="E.coli Chemotaxis Simulation" width="800"/>
  <p>Simulation of bacterial behavior algorithm (Reactive Robot)</p>

  [![EV3 Implementation](https://img.youtube.com/vi/UVyYjg4U8Js/0.jpg)](https://youtu.be/UVyYjg4U8Js)
  <p>Lego EV3 robot executing the algorithm</p>
</div>

### **Partial 2**
- **Perception And Localization**

We implemented **Markov Localization** to estimate the robot's position on a known map with unknown initial location. By updating a belief distribution based on sensor readings and movements, the robot gradually converges to its true position. The method was tested both in simulation and on a Lego EV3 robot using colored landmarks as references.

<div align="center">
  <img src="multimedia/animation_localization.gif" alt="Markov Localization Simulation" width="800"/>
  <p>Simulation of Markov Localization algorithm</p>

  [![EV3 Localization](https://img.youtube.com/vi/xWVI8FGqY8Y/0.jpg)](https://youtu.be/xWVI8FGqY8Y)
  <p>Lego EV3 robot executing the algorithm</p>
</div>


### **Partial 3**
- **Motion Planning**
- **Decision Making**

The objective is to simulate an intelligent robot navigating a known grid-world map. Initially, the robot
learns a policy to reach a fixed goal position using **Q-learning**. Later, the robot is placed at an unknown
location in the same map and must use **Markov Localization** to estimate its position. Once localization
confidence is high, the robot switches to the learned **Q-policy** to move toward the goal autonomously.

<div align="center">
  <img src="multimedia/learning_progress.gif" alt="Learning progress" width="800"/>
  <p>Learning progress of the robot over training episodes.</p>

  <img src="multimedia/confidence_animation.gif" alt="Confidence" width="800"/>
  <p>Localization confidence over time. The robot switches to the Q-policy after crossing the
threshold.</p>

  <img src="multimedia/belief_animation.gif" alt="Belief distribution" width="800"/>
  <p>Belief distribution over the grid. The cell with the highest probability (darkest blue)
indicates the robot’s estimated position.</p>

  <img src="multimedia/trajectory_animation.gif" alt="Robot Trajectory" width="800"/>
  <p>Robot trajectory (green = ground truth, blue = belief-based estimate).</p>
</div>


# Requirements
