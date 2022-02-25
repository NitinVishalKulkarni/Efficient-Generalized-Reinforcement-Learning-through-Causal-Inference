# Imports
import cv2
import gym
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import random


# Defining the Wumpus World Environment.
class WumpusWorldEnvironment(gym.Env):
    """This class implements the Wumpus World environment."""

    def __init__(self, environment_type):
        """This method initializes the environment.

        :param str environment_type: - (It can take two values: 1. 'training' 2. 'testing' indicating the type of
                                    environment.)"""

        self.environment_type = environment_type

        if self.environment_type == 'training':
            self.environment_width = 6
            self.environment_height = 6
            # This defines the total number of grid blocks in the environment.
            self.observation_space = spaces.Discrete(self.environment_width * self.environment_height)
            # This defines that there are 4 discrete actions that the agent can perform.
            self.action_space = spaces.Discrete(4)
            self.number_of_agents = 1  # This defines the number of agents in the environment.
            self.agent_pos = np.asarray([0, 0])  # This defines the agent's default initial position in the environment.
            # This defines the positions of breeze in the environment.
            self.breeze_pos = np.asarray([[1, 0], [3, 0], [5, 0], [2, 1], [4, 1], [1, 2], [3, 2], [5, 2], [0, 3],
                                          [2, 3], [1, 4], [3, 4], [5, 4], [0, 5], [2, 5], [4, 5]])
            self.gold_pos = np.asarray([4, 5])  # This defines the position of gold in the environment.
            self.gold_quantity = 1  # This defines the quantity of gold.
            # This defines the positions of pit in the environment.
            self.pit_pos = np.asarray([[2, 0], [5, 1], [2, 2], [0, 4], [2, 4], [3, 5], [5, 5]])
            # This defines the positions of stench in the environment.
            self.stench_pos = np.asarray([[3, 2], [2, 3], [4, 3], [3, 4]])
            self.wumpus_pos = np.asarray([3, 3])  # This defines the position of the Wumpus in the environment.
            self.timesteps = 0  # This defines the steps the agent has taken during an episode.
            self.max_timesteps = 1000  # This defines the maximum steps the agent can take during an episode.

            # Creating the mapping from the co-ordinates to the state.
            self.coordinates_state_mapping = {}
            for i in range(self.environment_height):
                for j in range(self.environment_width):
                    self.coordinates_state_mapping[f'{np.asarray([j, i])}'] = i * self.environment_width + j

            # Storing the terminal and non-terminal states.
            self.terminal_states = []
            self.non_terminal_states = []
            for position in self.coordinates_state_mapping:
                if np.array_equal(f'{self.wumpus_pos}', position) or np.array_equal(f'{self.gold_pos}', position) or \
                        any(np.array_equal(f'{self.pit_pos[i]}', position) for i in range(len(self.pit_pos))) or \
                        any(np.array_equal(f'{self.breeze_pos[i]}', position) for i in range(len(self.breeze_pos))) or \
                        any(np.array_equal(f'{self.stench_pos[i]}', position) for i in range(len(self.stench_pos))):
                    self.terminal_states.append(self.coordinates_state_mapping[position])
                else:
                    self.non_terminal_states.append(self.coordinates_state_mapping[position])

        elif self.environment_type == 'testing':
            self.environment_width = 6
            self.environment_height = 6
            # This defines the total number of grid blocks in the environment.
            self.observation_space = spaces.Discrete(self.environment_width * self.environment_height)
            # This defines that there are 4 discrete actions that the agent can perform.
            self.action_space = spaces.Discrete(4)
            self.number_of_agents = 1  # This defines the number of agents in the environment.
            self.agent_pos = np.asarray([0, 0])  # This defines the agent's default initial position in the environment.
            # This defines the positions of breeze in the environment.
            self.breeze_pos = np.asarray(
                [[1, 0], [3, 0], [5, 0], [2, 1], [4, 1], [1, 2], [3, 2], [5, 2], [0, 3], [2, 3],
                 [1, 4], [3, 4], [5, 4], [0, 5], [2, 5], [4, 5]])
            self.gold_pos = np.asarray([0, 5])  # This defines the position of gold in the environment.
            self.gold_quantity = 1  # This defines the quantity of gold.
            # This defines the positions of pit in the environment.
            self.pit_pos = np.asarray([[2, 0], [5, 1], [2, 2], [0, 4], [2, 4], [3, 5], [5, 5]])
            # This defines the positions of stench in the environment.
            self.stench_pos = np.asarray([[3, 2], [2, 3], [4, 3], [3, 4]])
            self.wumpus_pos = np.asarray([3, 3])  # This defines the position of the Wumpus in the environment.
            self.timesteps = 0  # This defines the steps the agent has taken during an episode.
            self.max_timesteps = 1000  # This defines the maximum steps the agent can take during an episode.

            # Creating the mapping from the co-ordinates to the state.
            self.coordinates_state_mapping = {}
            for i in range(self.environment_height):
                for j in range(self.environment_width):
                    self.coordinates_state_mapping[f'{np.asarray([j, i])}'] = i * self.environment_width + j

            # Storing the terminal and non-terminal states.
            self.terminal_states = []
            self.non_terminal_states = []
            for position in self.coordinates_state_mapping:
                if np.array_equal(f'{self.wumpus_pos}', position) or np.array_equal(f'{self.gold_pos}', position) or \
                        any(np.array_equal(f'{self.pit_pos[i]}', position) for i in range(len(self.pit_pos))) or \
                        any(np.array_equal(f'{self.breeze_pos[i]}', position) for i in range(len(self.breeze_pos))) or \
                        any(np.array_equal(f'{self.stench_pos[i]}', position) for i in range(len(self.stench_pos))):
                    self.terminal_states.append(self.coordinates_state_mapping[position])
                else:
                    self.non_terminal_states.append(self.coordinates_state_mapping[position])

    def partially_observable_state(self, agent_position):
        """This method returns the array to append to the states for partially observable MDP.
        :param arr agent_position: Integer representation of the state from the environment.

        :return: arr observation: Array representing the partial observation."""

        observation = np.zeros(9 * 5)
        positions_to_evaluate = [agent_position, [agent_position[0] - 1, agent_position[1]],
                                 [agent_position[0] - 1, agent_position[1] + 1],
                                 [agent_position[0], agent_position[1] + 1],
                                 [agent_position[0] + 1, agent_position[1] + 1],
                                 [agent_position[0] + 1, agent_position[1]],
                                 [agent_position[0] + 1, agent_position[1] - 1],
                                 [agent_position[0], agent_position[1] - 1],
                                 [agent_position[0] - 1, agent_position[1] - 1]]

        index = 0
        for position in positions_to_evaluate:
            if any(np.array_equal(position, self.breeze_pos[x]) for x in range(len(self.breeze_pos))):
                observation[index] = 1
            if any(np.array_equal(position, self.stench_pos[x]) for x in range(len(self.stench_pos))):
                observation[index + 1] = 1
            if any(np.array_equal(position, self.pit_pos[x]) for x in range(len(self.pit_pos))):
                observation[index + 2] = 1
            if np.array_equal(position, self.wumpus_pos):
                observation[index + 3] = 1
            if np.array_equal(position, self.gold_pos):
                observation[index + 4] = 1
            index += 5

        return observation

    def reset(self, random_start=False):
        """This method resets the agent position and returns the state as the observation.

        :param bool random_start: - Boolean indicating whether the agent will start in a random or fixed position.

        :returns arr observation: -  Array representing the partial observation."""

        if not random_start:
            self.agent_pos = np.asarray([1, 5])  # Upon resetting the environment the agent's position is set to [0, 0].
        else:
            # Randomly selecting the agent's position.
            random_state = random.choice(self.non_terminal_states)
            self.agent_pos = np.asarray([random_state % self.environment_width,
                                         int(np.floor(random_state / self.environment_width))])

        observation = self.partially_observable_state(self.agent_pos)
        self.timesteps = 0  # Resetting the number of steps taken by the agent.
        self.gold_quantity = 1  # Resetting the Gold quantity to be 1.

        return observation

    def step(self, action):
        """This method implements what happens when the agent takes a particular action. It changes the agent's
        position (While not allowing it to go out of the environment space.), maps the environment co-ordinates to a
        state, defines the rewards for the various states, and determines when the episode ends.

        :param int action: - Integer in the range 0 to 3 inclusive representing the different actions the agent can
        take.

        :returns arr observation: - Array representing the partial observation.
                 int reward: - Integer value that's used to measure the performance of the agent.
                 bool done: - Boolean describing whether the episode has ended.
                 dict info: - A dictionary that can be used to provide additional implementation information."""

        # Describing the outcomes of the various possible actions.
        if action == 0:
            self.agent_pos[0] += 1  # This action causes the agent to go right.
        if action == 1:
            self.agent_pos[0] -= 1  # This action causes the agent to go left.
        if action == 2:
            self.agent_pos[1] += 1  # This action causes the agent to go up.
        if action == 3:
            self.agent_pos[1] -= 1  # This action causes the agent to go down.

        # Ensuring that the agent doesn't go out of the environment.
        self.agent_pos = np.clip(self.agent_pos, a_min=[0, 0],
                                 a_max=[self.environment_width - 1, self.environment_height - 1])
        observation = self.partially_observable_state(self.agent_pos)

        self.timesteps += 1  # Increasing the total number of steps taken by the agent.

        reward = 0
        # Setting the reward to 10 if the agent reaches the gold.
        if np.array_equal(self.agent_pos, self.gold_pos) and self.gold_quantity > 0:
            self.gold_quantity -= 1
            reward = 1000

        for i in range(len(self.pit_pos)):  # Setting the reward to -1 if the agent falls in the pit.
            if np.array_equal(self.agent_pos, self.pit_pos[i]):
                reward = -50

        # Setting the reward to -1 if the agent is killed by Wumpus.
        if np.array_equal(self.agent_pos, self.wumpus_pos):
            reward = -100

        # The episode terminates when the agent reaches the Gold, or is killed by the Wumpus, falls into the pit, or
        # takes more than 10 steps.
        if self.gold_quantity == 0 or \
                np.array_equal(self.agent_pos, self.wumpus_pos):
            done = True
        else:
            done = False
        for i in range(len(self.pit_pos)):
            if np.array_equal(self.agent_pos, self.pit_pos[i]):
                done = True
        if self.timesteps == self.max_timesteps:
            done = True

        info = {}

        return observation, reward, done, info

    def render(self, mode='human', plot=False):
        """This method renders the environment.

        :param str mode: 'human' renders to the current display or terminal and returns nothing.
        :param bool plot: Boolean indicating whether we show a plot or not. If False, the method returns a resized NumPy
                     array representation of the environment to be used as the state. If True it plots the environment.

        :returns arr preprocessed_image: Grayscale NumPy array representation of the environment."""

        fig, ax = plt.subplots(figsize=(15, 15))  # Initializing the figure.
        ax.set_xlim(0, 6)  # Setting the limit on the x-axis.
        ax.set_ylim(0, 6)  # Setting the limit on the y-axis.

        def plot_image(plot_pos):
            """This is a helper function to render the environment. It checks which objects are in a particular
            position on the grid and renders the appropriate image.

            :param arr plot_pos: Co-ordinates of the grid position which needs to be rendered."""

            # Initially setting every object to not be plotted.
            plot_agent, plot_breeze, plot_gold, plot_pit, plot_stench, plot_wumpus = \
                False, False, False, False, False, False

            # Checking which objects need to be plotted by comparing their positions.
            if np.array_equal(self.agent_pos, plot_pos):
                plot_agent = True
            if any(np.array_equal(self.breeze_pos[i], plot_pos) for i in range(len(self.breeze_pos))):
                plot_breeze = True
            if self.gold_quantity > 0:  # Gold isn't plotted if it has already been picked by one of the agents.
                if np.array_equal(plot_pos, self.gold_pos):
                    plot_gold = True
            if any(np.array_equal(self.pit_pos[i], plot_pos) for i in range(len(self.pit_pos))):
                plot_pit = True
            if any(np.array_equal(self.stench_pos[i], plot_pos) for i in range(len(self.stench_pos))):
                plot_stench = True
            if np.array_equal(plot_pos, self.wumpus_pos):
                plot_wumpus = True

            # Plot for Agent.
            if plot_agent and \
                    all(not item for item in
                        [plot_breeze, plot_gold, plot_pit, plot_stench, plot_wumpus]):
                agent = AnnotationBbox(OffsetImage(plt.imread('./images/agent.png'), zoom=0.28),
                                       np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(agent)

            # Plot for Breeze.
            elif plot_breeze and \
                    all(not item for item in
                        [plot_agent, plot_gold, plot_pit, plot_stench, plot_wumpus]):
                breeze = AnnotationBbox(OffsetImage(plt.imread('./images/breeze.png'), zoom=0.28),
                                        np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(breeze)

            # Plot for Gold.
            elif plot_gold and \
                    all(not item for item in
                        [plot_agent, plot_breeze, plot_pit, plot_stench, plot_wumpus]):
                gold = AnnotationBbox(OffsetImage(plt.imread('./images/gold.png'), zoom=0.28),
                                      np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(gold)

            # Plot for Pit.
            elif plot_pit and \
                    all(not item for item in
                        [plot_agent, plot_breeze, plot_gold, plot_stench, plot_wumpus]):
                pit = AnnotationBbox(OffsetImage(plt.imread('./images/pit.png'), zoom=0.28),
                                     np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(pit)

            # Plot for Stench.
            elif plot_stench and \
                    all(not item for item in
                        [plot_agent, plot_breeze, plot_gold, plot_pit, plot_wumpus]):
                stench = AnnotationBbox(OffsetImage(plt.imread('./images/stench.png'), zoom=0.28),
                                        np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(stench)

            # Plot for Wumpus.
            elif plot_wumpus and \
                    all(not item for item in
                        [plot_agent, plot_breeze, plot_gold, plot_pit, plot_stench]):
                wumpus = AnnotationBbox(OffsetImage(plt.imread('./images/wumpus.png'), zoom=0.28),
                                        np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(wumpus)

            # Plot for Agent and Breeze.
            elif all(item for item in [plot_agent, plot_breeze]) and \
                    all(not item for item in
                        [plot_gold, plot_pit, plot_stench, plot_wumpus]):
                agent_breeze = AnnotationBbox(OffsetImage(plt.imread('./images/agent_breeze.png'), zoom=0.28),
                                              np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(agent_breeze)

            # Plot for Agent and Pit.
            elif all(item for item in [plot_agent, plot_pit]) and \
                    all(not item for item in
                        [plot_breeze, plot_gold, plot_stench, plot_wumpus]):
                agent_pit = AnnotationBbox(OffsetImage(plt.imread('./images/agent_dead_pit.png'), zoom=0.28),
                                           np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(agent_pit)

            # Plot for Agent and Stench.
            elif all(item for item in [plot_agent, plot_stench]) and \
                    all(not item for item in
                        [plot_breeze, plot_gold, plot_pit, plot_wumpus]):
                agent_stench = AnnotationBbox(OffsetImage(plt.imread('./images/agent_stench.png'), zoom=0.28),
                                              np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(agent_stench)

            # Plot for Agent, Breeze and Stench.
            elif all(item for item in [plot_agent, plot_breeze, plot_stench]) and \
                    all(not item for item in
                        [plot_gold, plot_pit, plot_wumpus]):
                agent_breeze_stench = AnnotationBbox(OffsetImage(plt.imread('./images/agent_breeze_stench.png'),
                                                                 zoom=0.28), np.add(plot_pos, [0.5, 0.5]),
                                                     frameon=False)
                ax.add_artist(agent_breeze_stench)

            # Plot for Agent and Wumpus.
            elif all(item for item in [plot_agent, plot_wumpus]) and \
                    all(not item for item in
                        [plot_gold, plot_pit, plot_stench, plot_breeze]):
                agent_wumpus = AnnotationBbox(OffsetImage(plt.imread('./images/agent_dead_wumpus_alive.png'),
                                                          zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(agent_wumpus)

            # Plot for Breeze and Gold.
            elif all(item for item in [plot_breeze, plot_gold]) and \
                    all(not item for item in
                        [plot_agent, plot_pit, plot_stench, plot_wumpus]):
                breeze_gold = AnnotationBbox(OffsetImage(plt.imread('./images/breeze_gold.png'), zoom=0.28),
                                             np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(breeze_gold)

            # Plot for Breeze and Stench.
            elif all(item for item in [plot_breeze, plot_stench]) and \
                    all(not item for item in
                        [plot_agent, plot_gold, plot_pit, plot_wumpus]):
                breeze_stench = AnnotationBbox(OffsetImage(plt.imread('./images/breeze_stench.png'), zoom=0.28),
                                               np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(breeze_stench)

            # Plot for Breeze, Stench, and Gold.
            elif all(item for item in [plot_breeze, plot_gold, plot_stench]) and \
                    all(not item for item in
                        [plot_agent, plot_pit, plot_wumpus]):
                breeze_gold_stench = AnnotationBbox(OffsetImage(plt.imread('./images/breeze_gold_stench.png'),
                                                                zoom=0.28), np.add(plot_pos, [0.5, 0.5]),
                                                    frameon=False)
                ax.add_artist(breeze_gold_stench)

            # Plot for Stench and Gold.
            elif all(item for item in [plot_stench, plot_gold]) and \
                    all(not item for item in
                        [plot_agent, plot_breeze, plot_pit, plot_wumpus]):
                stench_gold = AnnotationBbox(OffsetImage(plt.imread('./images/stench_gold.png'), zoom=0.28),
                                             np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(stench_gold)

        coordinates_state_mapping_2 = {}
        for j in range(self.environment_height * self.environment_width):
            coordinates_state_mapping_2[j] = np.asarray(
                [j % self.environment_width, int(np.floor(j / self.environment_width))])

        # Rendering the images for all states.
        for position in coordinates_state_mapping_2:
            plot_image(coordinates_state_mapping_2[position])

        plt.xticks([0, 1, 2, 3, 4, 5])  # Specifying the ticks on the x-axis.
        plt.yticks([0, 1, 2, 3, 4, 5])  # Specifying the ticks on the y-axis.
        plt.grid()  # Setting the plot to be of the type 'grid'.

        if plot:  # Displaying the plot.
            plt.show()
        else:  # Returning the preprocessed image representation of the environment.
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
            # width = int(img.shape[1] * 84 / 1000)
            # height = int(img.shape[0] * 84 / 1000)
            width = img.shape[1]
            height = img.shape[0]
            dim = (width, height)
            # noinspection PyUnresolvedReferences
            preprocessed_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            plt.show()
            return preprocessed_image
