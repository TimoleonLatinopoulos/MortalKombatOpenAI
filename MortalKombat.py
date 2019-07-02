import gym
import numpy as np
import retro
from cv2 import flip

from Utilities import PreprocessFrame


class MultiDiscretizer(gym.ActionWrapper):

    def __init__(self, env):
        super(MultiDiscretizer, self).__init__(env)
        self.buttons = ["B", "Y", "NOOP", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
        actions_1 = [['NOOP'], ['UP'], ['DOWN']]
        actions_2 = [['NOOP'], ['LEFT'], ['RIGHT']]
        actions_3 = [['NOOP'], ['A'], ['B'], ['X'], ['Y'], ['L']]
        self._actions1 = self.one_hot(actions_1)
        self._actions2 = self.one_hot(actions_2)
        self._actions3 = self.one_hot(actions_3)

        self.action_space = gym.spaces.MultiDiscrete([len(self._actions1), len(self._actions2), len(self._actions3)])

    def one_hot(self, actions):
        _actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[self.buttons.index(button)] = True
            _actions.append(arr)
        return _actions

    def action(self, a):  # pylint: disable=W0221
        return [x or y or z for (x, y, z) in zip(self._actions1[a[0]], self._actions2[a[1]], self._actions3[a[2]])]


class Frameskip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def reset(self):
        return self.env.reset()

    def step(self, act):
        total_rew = 0.0
        done = None
        for i in range(self._skip):
            obs, rew, done, info = self.env.step(act)
            total_rew += rew
            if done:
                break
        return obs, total_rew, done, info


class Environment:
    """ Manages the game by using the Retro Environment which provides all the information needed """

    def __init__(self, game, state, scenario, frame_height, frame_width, frame_skip):
        self.env = retro.make(game=game, state=state, scenario=scenario)
        self.env = MultiDiscretizer(self.env)
        self.frame_skip = frame_skip
        # self.env = Frameskip(self.env, frame_skip)
        self.frame_processor = PreprocessFrame(frame_height, frame_width)
        self.state = None
        self.p1_current_wins = 0
        self.p2_current_wins = 0

    # Resets the game environment
    def reset(self, sess, stacked_frames):
        observation = self.env.reset()
        self.p1_current_wins = 0
        self.p2_current_wins = 0

        processed_observation = self.frame_processor.process(sess, observation)
        self.state = np.repeat(processed_observation, stacked_frames, axis=2)

    # Applies the action given to the environment and returns all the current frame,
    # reward and whether the match has finished. It also saves the current state of
    # the game that will be used to get the next action
    def step(self, sess, action, p_info, evaluation=False):

        if p_info and p_info['p1_X'] - p_info['p2_X'] > 0:
            if action[1] == 2:
                action[1] = 1
            elif action[1] == 1:
                action[1] = 2

        observation, reward, done, info = self.env.step(action)

        if info['p1_X'] - info['p2_X'] > 0:
            observation = flip(observation, 1)

        p1_round_wins = info['p1_round_wins']
        p2_round_wins = info['p2_round_wins']

        if p1_round_wins > self.p1_current_wins:
            self.p1_current_wins = p1_round_wins
            # reward += 100.0
        if p2_round_wins > self.p2_current_wins:
            self.p2_current_wins = p2_round_wins
            # reward -= 100.0

        if not evaluation:
            if p1_round_wins == 1 or p2_round_wins == 1:
                done = True

        processed_observation = self.frame_processor.process(sess, observation)
        new_state = np.append(self.state[:, :, 1:], processed_observation, axis=2)
        self.state = new_state

        return processed_observation[:, :, 0], observation, info, reward, done
