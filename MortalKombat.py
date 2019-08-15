import gym
import numpy as np
import retro
from cv2 import flip, Canny

from Utilities import PreprocessFrame


class Discretizer(gym.ActionWrapper):
    """ Converts the actions available for the game environment to 12 discrete possible actions """

    def __init__(self, env):
        super(Discretizer, self).__init__(env)
        self.buttons = ["B", "Y", "NOOP", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
        actions = [['NOOP'], ['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], ['L'], ['DOWN', 'A'], ['DOWN', 'B'],
                   ['RIGHT', 'A'], ['LEFT', 'A'], ['LEFT', 'B'], ['RIGHT', 'B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[self.buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


class Frameskip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def reset(self):
        return self.env.reset()

    def step(self, act):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            observation, reward, done, info = self.env.step(act)
            total_reward += reward
            if done:
                break
        return observation, total_reward, done, info


class Environment:
    """ Manages the game by using the Retro Environment which provides all the information needed """

    def __init__(self, game, state, scenario, frame_height, frame_width, frame_skip):
        self.env = retro.make(game=game, state=state, scenario=scenario)
        self.env = Discretizer(self.env)
        self.frame_skip = frame_skip
        self.env = Frameskip(self.env, frame_skip)
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
            if action == 3:
                action = 4
            elif action == 4:
                action = 3
            elif action == 8:
                action = 9
            elif action == 9:
                action = 8
            elif action == 10:
                action = 11
            elif action == 11:
                action = 10

        observation, reward, done, info = self.env.step(action)

        if info['p1_X'] - info['p2_X'] > 0:
            observation = flip(observation, 1)

        p1_round_wins = info['p1_round_wins']
        p2_round_wins = info['p2_round_wins']

        if p1_round_wins > self.p1_current_wins:
            self.p1_current_wins = p1_round_wins
        if p2_round_wins > self.p2_current_wins:
            self.p2_current_wins = p2_round_wins

        if info['p1_hp'] != 0 and info['p2_hp'] != 0:
            reward = info['p1_hp'] - info['p2_hp']

        if not evaluation:
            if p1_round_wins == 1 or p2_round_wins == 1:
                done = True

        if format(info['timer'], '02x') == '00' and info['p1_hp'] == info['p2_hp']:
                done = True

        processed_observation = self.frame_processor.process(sess, observation)
        processed_observation = Canny(processed_observation, 100, 200)[:, :, np.newaxis]
        new_state = np.append(self.state[:, :, 1:], processed_observation, axis=2)
        self.state = new_state

        return processed_observation[:, :, 0], observation, info, reward, done
