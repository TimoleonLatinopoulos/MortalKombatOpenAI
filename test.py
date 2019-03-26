import retro
import gym
import numpy as np

GAME = 'MortalKombat-SNES'
STATES = ['Level1.SubZeroVsJohnnyCage', 'Level2', 'Level3', 'Level4', 'Level5', 'Level6', 'Level7']


class Discretizer(gym.ActionWrapper):

    def __init__(self, env):
        super(Discretizer, self).__init__(env)
        buttons = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
        actions = [['A'], ['B'], ['X'], ['Y'], ['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], ['L']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()


def main():
    env = Discretizer(retro.make(game=GAME, state=STATES[0]))

    # print("The environment has the following {} actions: {}".format(env.action_space.n, env.buttons))
    print("The environment has {} actions.".format(env.action_space.n))

    obs = env.reset()

    done = False
    p1_current_wins = 0
    p2_current_wins = 0
    end_reward = 0

    while not done:
        env.render()

        action = env.action_space.sample()
        # print(action)
        obs, reward, done, info = env.step(action)

        p1_round_wins = info['p1_round_wins']
        p2_round_wins = info['p2_round_wins']

        if p1_round_wins > p1_current_wins:
            p1_current_wins = p1_round_wins
            reward += 100.0
        if p2_round_wins > p2_current_wins:
            p2_current_wins = p2_round_wins
            reward -= 100.0

        end_reward += reward
        if reward != 0.0:
            print(reward, end_reward)

    print("\nEnd of fight. Reward: {}".format(end_reward))


if __name__ == "__main__":
    main()
