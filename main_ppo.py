import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces

class CropRowEnv(gym.Env):
    """
    Custom Gym environment for crop row path planning with two-component action:
      - The first element controls the robot’s orientation (0 for upward, 1 for downward).
      - The second element controls the movement command:
            * If < 2, it is a vertical move (0 = forward, 1 = backward).
            * If >= 2, it indicates a corridor-switching action.
    The observation is [corridor, vertical_position, orientation, driving_direction, sampling_x, sampling_y].
    A small extra penalty is given when the agent oscillates between forward and backward.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 num_crop_rows=5,      # Number of crop rows (vertical lines). Corridors = num_crop_rows - 1.
                 corridor_length=12,   # Vertical positions: -1 to 11.
                 max_episode_steps=100
                ):
        super(CropRowEnv, self).__init__()

        self.max_episode_steps = max_episode_steps
        self.num_crop_rows = num_crop_rows
        self.num_corridors = num_crop_rows - 1  # corridors between crop rows
        self.corridor_length = corridor_length + 2  # add 2 as before

        # Set the observation space:
        # [corridor, vertical_position, orientation, driving_direction, sampling_x, sampling_y]
        self.observation_space = spaces.Box(
            low=np.array([0.5, -1.0, -1.0, 0.5, 0, 0], dtype=np.float32),
            high=np.array([
                self.num_corridors - 0.5,
                self.corridor_length - 1,
                1.0,  # orientation: 0 or 1
                self.num_corridors - 0.5,    # target corridor
                self.num_crop_rows - 1,
                self.corridor_length - 2
            ], dtype=np.float32),
            dtype=np.float32
        )

        # Use a MultiDiscrete action space with 2 elements:
        # First element: orientation command, with 2 options (0 or 1)
        # Second element: movement command, with (2 + num_corridors) options
        self.action_space = spaces.MultiDiscrete([2, 2 + self.num_corridors])

        # These variables will be set by the agent:
        self.orientation = None      # fixed robot orientation (0: upward, 1: downward)
        self.drivingd = None         # vertical move command: 0 for forward, 1 for backward

        # Other environment variables:
        self.state = None
        self.sampling_point = None
        self.path = []  # store robot positions (corridor, pos)
        self.fig, self.ax = plt.subplots()
        self.total_reward = 0
        self.turn = False

        # To help penalize oscillations:
        self.previous_action = None

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        self.current_step = 0
        if seed is not None:
            np.random.seed(seed)

        # Randomize the start position
        self.state = (np.random.randint(self.num_corridors - 1) + 0.5,
                      np.random.randint(0, self.corridor_length - 2))
        # Randomize the goal position
        self.sampling_point = (np.random.randint(self.num_crop_rows),
                               np.random.randint(0, self.corridor_length - 2))
        # For the new episode, leave orientation and driving direction undefined.
        self.orientation = None
        self.drivingd = None

        self.path = [self.state]
        self.total_reward = 0
        self.previous_action = None

        # Return observation. If orientation or driving direction are not set, use -1.
        return np.array([
            self.state[0],
            self.state[1],
            -1.0,      # orientation undefined
            self.state[0],        # driving direction undefined
            self.sampling_point[0],
            self.sampling_point[1]
        ], dtype=np.float32), {}

    def _get_robot_coords(self):
        """Return the robot's current (x, y) coordinates for visualization."""
        corridor, pos = self.state
        return (corridor + 0.5, pos)

    def step(self, action):
        """Execute an action. The action is an array: [ori_action, move_action]."""
        self.current_step += 1
        corridor, pos = self.state
        self.current_corridor = corridor

        # Unpack action: 
        # The first element specifies the robot's orientation (0: upward, 1: downward).
        # The second element specifies the movement command:
        #   <2: vertical move (0: forward, 1: backward)
        #   >=2: corridor switching
        ori_action, move_action = action

        # Override the environment's orientation with the agent's command.
        # self.orientation = ori_action

        # Default step penalty
        reward = -0.1  
        done = False
        truncated = False

        # Check if at an end of the corridor.
        at_end = (pos == -1 or pos == self.corridor_length - 1)
        # print("orientation", ori_action, self.orientation)
        if not at_end and self.orientation is not None and ori_action != self.orientation:
            reward -= 1.0  # Penalize changing orientation within a row.
        # print("reward", reward)
        # Apply the chosen orientation
        self.orientation = ori_action

        # Penalize oscillatory vertical moves (if not at the boundary)
        if not at_end and self.previous_action is not None:
            # Compare the previous move_action with the current one.
            prev_move = self.previous_action[1]
            if (move_action == 0 and prev_move == 1) or (move_action == 1 and prev_move == 0):
                reward -= 1.0
        print("reward", reward)
        # Interpret the move_action:
        if move_action < 2:
            # Vertical movement.
            self.drivingd = move_action
        elif move_action >= 2:
            # Corridor switching command.
            value = move_action - 1.5  # target corridor index
        else:
            reward = -0.6

        # Vertical movement (if move_action is 0 or 1):
        if move_action < 2:
            if self.drivingd == 0:  # forward
                if self.orientation == 0:  # robot facing up: forward means increasing pos
                    if pos < self.corridor_length - 1:
                        pos += 1
                    else:
                        reward = -0.5
                elif self.orientation == 1:  # robot facing down: forward means decreasing pos
                    if pos > -1:
                        pos -= 1
                    else:
                        reward = -0.5
            elif self.drivingd == 1:  # backward
                if self.orientation == 0:  # robot facing up: backward means moving down
                    if pos > -1:
                        pos -= 1
                    else:
                        reward = -0.5
                elif self.orientation == 1:  # robot facing down: backward means moving up
                    if pos < self.corridor_length - 1:
                        pos += 1
                    else:
                        reward = -0.5

        # Corridor switching: if move_action >= 2.
        elif move_action >= 2:
            if at_end:
                if 0.5 <= value < self.num_corridors:
                    corridor = value
                    self.turn = True
                    # When switching corridors, the agent will select new orientation in the next step.
                    # We set drivingd to None so the agent must choose a new movement command.
                    self.drivingd = None
                    reward = -0.1
                else:
                    reward = -0.5
            else:
                reward = -0.6

        # Update previous_action with the current action array.
        self.previous_action = action

        self.state = (corridor, pos)
        self.path.append(self.state)

        # Determine the crop row on the robot's left.
        # For a fixed orientation, if orientation==0 (up) then left_crop_row = corridor - 0.5; if 1 then corridor + 0.5.
        left_crop_row = None
        if self.orientation is not None:
            if self.orientation == 0:
                left_crop_row = corridor - 0.5
            elif self.orientation == 1:
                left_crop_row = corridor + 0.5

        # Check goal condition.
        goal_crop, goal_pos = self.sampling_point
        if (pos == goal_pos) and (left_crop_row is not None) and (left_crop_row == goal_crop):
            reward += 10.0
            done = True

        self.total_reward += reward
        obs = np.array([
            corridor, pos,
            self.orientation if self.orientation is not None else -1.0,
            corridor,
            self.sampling_point[0], self.sampling_point[1]
        ], dtype=np.float32)
        if not done and self.current_step >= self.max_episode_steps:
            truncated = True

        return obs, reward, done, truncated, {}

    def render(self, mode="human"):
        """Render the environment with the robot's full path."""
        self.ax.clear()
        # Draw crop rows as vertical lines.
        for i in range(self.num_crop_rows):
            self.ax.plot([i, i], [0.0, self.corridor_length - 2], color='green', linewidth=2)
        self.ax.set_xlim(-0.5, self.num_crop_rows)
        self.ax.set_ylim(-1.5, self.corridor_length - 0.5)
        self.ax.set_xlabel("Crop Rows")
        self.ax.set_ylabel("Position along corridor")
        self.ax.set_title("Crop Row Path Planning Environment")
        self.ax.set_aspect('equal')

        if len(self.path) > 1:
            xs, ys = zip(*self.path)
            self.ax.plot(xs, ys, '--', color='orange', linewidth=1, label="Path")

        robot_x, robot_y = self.state
        self.ax.plot(robot_x, robot_y, 'ro', markersize=12, label="Robot")

        # Draw orientation arrow (blue) based solely on the agent's chosen orientation.
        if self.orientation is not None:
            if self.orientation == 0:
                self.ax.arrow(robot_x, robot_y, -0.4, 0, head_width=0.2, head_length=0.2, fc='b', ec='b')
                self.ax.arrow(robot_x, robot_y, 0, 0.4, head_width=0.2, head_length=0.2, fc='b', ec='b')
            elif self.orientation == 1:
                self.ax.arrow(robot_x, robot_y, 0.4, 0, head_width=0.2, head_length=0.2, fc='b', ec='b')
                self.ax.arrow(robot_x, robot_y, 0, -0.4, head_width=0.2, head_length=0.2, fc='b', ec='b')

        goal_x = self.sampling_point[0]
        goal_y = self.sampling_point[1]
        self.ax.plot(goal_x, goal_y, 'b*', markersize=15, label="Goal")

        self.ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.pause(0.5)

    def close(self):
        plt.close()

if __name__ == "__main__":
    num_crop_rows = 5
    corridor_length = 5
    env = CropRowEnv(num_crop_rows=num_crop_rows, corridor_length=corridor_length)
    state = env.reset()
    print("Initial state:", state, "Orientation:", env.orientation, "Goal:", env.sampling_point)
    print("Accumulated Reward: {:.2f}".format(env.total_reward))
    
    done = False
    while not done:
        # For the first step, if drivingd is not set, choose a random vertical action for the second element.
        if env.drivingd is None:
            move_action = np.random.choice([0, 1])
        else:
            # Otherwise, if not at the boundary, just repeat the previous move action.
            if env.state[1] not in (-1, env.corridor_length - 1):
                move_action = env.previous_action[1] if env.previous_action is not None else np.random.choice([0, 1])
            else:
                # At boundary, switch corridor: choose a random corridor switching action.
                possible_corridors = [i + 0.5 for i in range(env.num_corridors)]
                target = np.random.choice(possible_corridors)
                move_action = int(2 + target)
        # For the orientation element, you can either keep the current orientation or choose a new one.
        # Here, we let the agent decide by sampling randomly for testing.
        ori_action = np.random.choice([0, 1])
        action = np.array([ori_action, move_action])
        
        state, reward, done, _, _ = env.step(action)
        print(f"Action: {action} -> State: {state}, Orientation: {env.orientation}, Step Reward: {reward:.2f}, Accumulated Reward: {env.total_reward:.2f}, Done: {done}")
        env.render()
        
    env.close()
    print("Episode finished with total reward: {:.2f}".format(env.total_reward))
