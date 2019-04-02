import retro
import gym
import numpy as np
from Utilities import PreprocessFrame


class Discretizer(gym.ActionWrapper):

    def __init__(self, env):
        super(Discretizer, self).__init__(env)
        buttons = ['B', 'Y', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'X', 'L', 'R']
        actions = [['A'], ['B'], ['X'], ['Y'], ['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], ['L']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


class Environment:

    def __init__(self, game, state, frame_height, frame_width):
        self.env = Discretizer(retro.make(game=game, state=state))
        self.frame_processor = PreprocessFrame(frame_height, frame_width)
        self.state = None
        self.p1_current_wins = 0
        self.p2_current_wins = 0

    def reset(self, sess, stacked_frames):
        observation = self.env.reset()
        self.p1_current_wins = 0
        self.p2_current_wins = 0

        processed_observation = self.frame_processor.process(sess, observation)
        self.state = np.repeat(processed_observation, stacked_frames, axis=2)

    def step(self, sess, action):

        observation, reward, done, info = self.env.step(action)

        p1_round_wins = info['p1_round_wins']
        p2_round_wins = info['p2_round_wins']

        if p1_round_wins > self.p1_current_wins:
            self.p1_current_wins = p1_round_wins
            reward += 100.0
        if p2_round_wins > self.p2_current_wins:
            self.p2_current_wins = p2_round_wins
            reward -= 100.0

        processed_observation = self.frame_processor.process(sess, observation)
        new_state = np.append(self.state[:, :, 1:], processed_observation, axis=2)
        self.state = new_state

        return processed_observation[:, :, 0], observation, reward, done
