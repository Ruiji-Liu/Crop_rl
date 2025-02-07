import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces


class CropRowEnv(gym.Env):
    """
    Custom Gym environment for crop row path planning with orientation, sampling on crop rows,
    and path visualization.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 num_crop_rows=5,      # Number of crop rows (vertical lines). Corridors = num_crop_rows - 1.
                 corridor_length=12    # Vertical positions: -1 to 11.
                ):
        super(CropRowEnv, self).__init__()

        self.num_crop_rows = num_crop_rows
        self.num_corridors = num_crop_rows - 1  # corridors between crop rows
        self.corridor_length = corridor_length

        # Randomize start position (corridor and vertical position)
        self.start_state = (np.random.randint(self.num_corridors),
                            np.random.randint(0, self.corridor_length - 2))
        self.state = self.start_state

        # Randomize goal (sampling point)
        self.sampling_point = (np.random.randint(self.num_crop_rows),
                               np.random.randint(0, self.corridor_length - 2))
        # The robot's state: (corridor, vertical_position)
        self.observation_space = spaces.Box(
            low=np.array([0, -1.5], dtype=np.float32),
            high=np.array([self.num_corridors - 1, self.corridor_length - 0.5], dtype=np.float32),
            dtype=np.float32
        )

        # Action space: MultiDiscrete with two components.
        # self.action_space = spaces.MultiDiscrete([3, self.num_corridors])
        # self.action_space = spaces.Discrete(3 * self.num_corridors)
        self.action_space = spaces.Discrete(2 + self.num_corridors)
        # Orientation: None until a vertical move is initiated.
        # 0: up (left is west; crop row = corridor)
        # 1: down (left is east; crop row = corridor + 1)
        self.orientation = None

        #initial corridor
        self.initial_corridor = True

        # Store the robot's path as a list of (x, y) positions.
        self.path = []

        # For visualization.
        self.fig, self.ax = plt.subplots()

        # Accumulated reward (total reward for the episode)
        self.total_reward = 0

        # Whether the robot turns
        self.turn = False

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        # Reset the random seed if provided
        if seed is not None:
            np.random.seed(seed)
        # Here you can randomize the state or set fixed values for testing
        self.state = (np.random.randint(self.num_corridors),
                            np.random.randint(0, self.corridor_length - 2))
        # self.state = (3, 8)
        self.orientation = None
        self.sampling_point = (np.random.randint(self.num_crop_rows),
                               np.random.randint(0, self.corridor_length - 2))
        # self.sampling_point = (3,3)
        self.path = []
        self.path.append(self._get_robot_coords())
        self.total_reward = 0
        return np.array(self.state, dtype=np.int32)

    def _get_robot_coords(self):
        """Return the robot's current (x, y) coordinates for visualization."""
        corridor, pos = self.state
        # The x-coordinate is corridor index plus 0.5 (center of corridor).
        return (corridor + 0.5, pos)

    def step(self, action):
        """
        Execute an action.
        Action is a two-element array: [action_type, value].
        Returns: (observation, step_reward, done, info)
        """
        corridor, pos = self.state
        # action_type, value = action
        # action_type = action // self.num_corridors
        # value = action % self.num_corridors
        if action < 2:
            action_type = 0
            value = action  # 0: up, 1: down
        else:
            action_type = 2
            value = action - 2 
        reward = -0.1  # Default step penalty
        done = False

        # Check if at an end of the corridor.
        at_end = (pos == -1 or pos == self.corridor_length - 1)
        print("in step function, action type is", action_type)
        if action_type == 0:
            self.turn = False
            print("in action type 0")
            # Vertical movement.
            # if self.initial_corridor is None:
            #     self.initial_corridor = True

            
            if self.orientation is None:
                # print("orientation is None", value)
                if value == 0:
                    self.orientation = 0  # up
                elif value == 1:
                    self.orientation = 1  # down
                else:
                    reward = -0.6  # invalid value
            print("orientation", self.orientation)

            if self.orientation == 0:  # moving up
                if pos < self.corridor_length - 1:
                    pos += 1
            elif self.orientation == 1:  # moving down
                if pos > -1:
                    pos -= 1
        elif action_type == 2:
            # Switch corridor.
            # print("at end")
            self.initial_corridor = False
            if at_end:
                if 0 <= value < self.num_corridors and value != corridor:
                    corridor = value
                    print("pos", pos, self.corridor_length - 1)
                    if pos == self.corridor_length - 1:
                        self.orientation = 1
                    else:
                        self.orientation = 0  # reset orientation after switching
                    self.turn = True
                    # (Optional) You could penalize a turn as well here

                    reward -= 0.3  # Penalize switching/turning too frequently
                else:
                    reward = -0.5  # invalid target corridor
            else:
                reward = -0.5  # switching not allowed if not at an end
        else:
            reward = -0.5  # unknown action

        self.state = (corridor, pos)
        self.path.append(self._get_robot_coords())

        # Determine the crop row on the robot's left.
        left_crop_row = None
        if self.orientation is not None:
            if self.initial_corridor:
                left_crop_row = corridor
            if self.orientation == 0 and self.initial_corridor is False:  # up: left is the crop row at index = corridor
                left_crop_row = corridor
            elif self.orientation == 1 and self.initial_corridor is False:  # down: left is the crop row at index = corridor + 1
                left_crop_row = corridor + 1

        # Check goal condition.
        goal_crop, goal_pos = self.sampling_point
        if (pos == goal_pos) and (left_crop_row is not None) and (left_crop_row == goal_crop):
            reward += 10.0
            done = True

        # Accumulate the reward (for monitoring purposes)
        self.total_reward += reward

        # Return the immediate reward (for DQN training) along with observation and done flag.
        return np.array(self.state, dtype=np.int32), reward, done, {}

    def render(self, mode="human"):
        """Render the environment with the robot's full path."""
        self.ax.clear()
        # Draw crop rows as vertical lines from 0 to (num_crop_rows - 1)
        for i in range(self.num_crop_rows):
            self.ax.plot([i, i], [0.0, self.corridor_length - 2], color='green', linewidth=2)
        self.ax.set_xlim(-0.5, self.num_crop_rows)
        self.ax.set_ylim(-1.5, self.corridor_length - 0.5)
        self.ax.set_xlabel("Crop Rows")
        self.ax.set_ylabel("Position along corridor")
        self.ax.set_title("Crop Row Path Planning Environment")
        self.ax.set_aspect('equal')

        # Draw the robot's path.
        if len(self.path) > 1:
            xs, ys = zip(*self.path)
            self.ax.plot(xs, ys, 'k--', color='orange', linewidth=1, label="Path")

        # Draw the current robot position.
        robot_x, robot_y = self._get_robot_coords()
        self.ax.plot(robot_x, robot_y, 'ro', markersize=12, label="Robot")

        # Draw orientation arrows.
        if self.orientation is not None:
            if self.initial_corridor is True:
                # print("here1")
                self.ax.arrow(robot_x, robot_y, -0.4, 0, head_width=0.2, head_length=0.2, fc='r', ec='r')
                self.ax.arrow(robot_x, robot_y, 0, 0.4, head_width=0.2, head_length=0.2, fc='b', ec='b')
            elif self.orientation == 0 and self.initial_corridor is False:
                # print("here2")
                self.ax.arrow(robot_x, robot_y, -0.4, 0, head_width=0.2, head_length=0.2, fc='r', ec='r')
                self.ax.arrow(robot_x, robot_y, 0, 0.4, head_width=0.2, head_length=0.2, fc='b', ec='b')
            elif self.orientation == 1 and self.initial_corridor is False:
                # print("here3")
                # Right arrow (red) and forward arrow (blue) when facing down.
                self.ax.arrow(robot_x, robot_y, 0.4, 0, head_width=0.2, head_length=0.2, fc='r', ec='r')
                self.ax.arrow(robot_x, robot_y, 0, -0.4, head_width=0.2, head_length=0.2, fc='b', ec='b')

        # Draw the sampling point (goal) on its crop row.
        goal_x = self.sampling_point[0]
        goal_y = self.sampling_point[1]
        self.ax.plot(goal_x, goal_y, 'b*', markersize=15, label="Goal")

        self.ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.pause(0.1)

    def close(self):
        plt.close()
if __name__ == "__main__":
    num_crop_rows = 10
    corridor_length = 10
    env = CropRowEnv(num_crop_rows=num_crop_rows, corridor_length=corridor_length + 2)
    state = env.reset()
    print("Initial state:", state, "Orientation:", env.orientation, "Goal:", env.sampling_point)
    print("Accumulated Reward: {:.2f}".format(env.total_reward))
    
    done = False
    while not done:
        # If orientation is not yet set, initialize with a vertical action
        if env.orientation is None and env.initial_corridor is True:
            random_choice = np.random.choice([0, 1])
            action = random_choice  # Randomly choose up (0) or down (1) to set orientation
        else:
            # If not at the boundary, keep moving vertically
            if env.state[1] not in (-1, env.corridor_length - 1):
                action = 0  # Move up
            else:
                # At boundary, switch corridor
                if env.turn:
                    action = env.orientation  # Continue in the same orientation
                else:
                    possible_corridors = [c for c in range(env.num_corridors) if c != env.state[0]]
                    target = np.random.choice(possible_corridors) if possible_corridors else env.state[0]
                    action = 2 + target  # Switch to a new corridor (action >= 2)
        
        state, reward, done, _ = env.step(action)
        print(f"Action: {action} -> State: {state}, Step Reward: {reward:.2f}, Accumulated Reward: {env.total_reward:.2f}, Done: {done}")
        env.render()
        
    env.close()
    print("Episode finished with total reward: {:.2f}".format(env.total_reward))

# if __name__ == "__main__":
#     num_crop_rows = 10
#     corridor_length = 10
#     env = CropRowEnv(num_crop_rows=num_crop_rows, corridor_length=corridor_length + 2)
#     state = env.reset()
#     print("Initial state:", state, "Orientation:", env.orientation, "Goal:", env.sampling_point)
#     print("Accumulated Reward: {:.2f}".format(env.total_reward))
    
#     done = False
#     while not done:
#         # If orientation is not yet set, initialize with a vertical action
#         print("orientation", env.orientation)
#         if env.orientation is None and env.initial_corridor is True:
#             random_choice = np.random.choice([0, 1])
#             print ("random choice", random_choice)
#             action = [0, random_choice]  # Randomly choose up or down to set orientation
#             # action = [0, 0]
        
#         else:
#             print("initial corridor", env.initial_corridor)
            
#             # If not at the boundary, keep moving vertically
#             if env.state[1] not in (-1, env.corridor_length - 1):
#                 print("keep moving")
#                 action = [0, 0]
#             else:
#                 # At boundary, switch corridor
#                 if env.turn:
#                     action = [0, env.orientation]
#                 else:
#                     possible_corridors = [c for c in range(env.num_corridors) if c != env.state[0]]
#                     target = np.random.choice(possible_corridors) if possible_corridors else env.state[0]
#                     action = [2, target]
#             # if env.initial_corridor is False:
#             #     print("call")
#             #     print("orientation", env.orientation)
#             #     action = [0, env.orientation]
#         state, reward, done, _ = env.step(action)
#         # print(f"Action: {action} -> State: {state}, Step Reward: {reward:.2f}, Accumulated Reward: {env.total_reward:.2f}, Done: {done}")
#         env.render()
        
#     env.close()
#     print("Episode finished with total reward: {:.2f}".format(env.total_reward))
