import random
from os import fsync

import imageio
import numpy as np
import tensorflow as tf


# Generates a gif from an array of frames and saves it to the path given
def generate_gif(episodes, reward, result, frames, path):
    for i, frame in enumerate(frames):
        frames[i] = frame
    pathname = path + "Episode " + str(episodes) + "  Reward " + str(reward) + " " + result + " " + ".gif"
    imageio.mimsave(pathname, frames, duration=1 / 60)


class ReplayMemory:
    """ Stores transitions of states from the Retro Environment """

    def __init__(self, frame_height=63, frame_width=113, stacked_frames=4, batch_size=32, memory_size=900000,
                 load_memory=False):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.stacked_frames = stacked_frames
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.file_names = {"observations", "actions", "rewards", "done", "count", "current", "states", "new_states",
                           "indices"}

        if load_memory:
            for file_name in self.file_names:
                setattr(self, file_name, np.load("output/" + file_name + ".npy"))
        else:
            self.count = 0
            self.current = 0

            # Pre-allocate memory
            self.observations = np.empty((self.memory_size, self.frame_height, self.frame_width), dtype=np.uint8)
            self.actions = np.empty(self.memory_size, dtype=np.uint8)
            self.rewards = np.empty(self.memory_size, dtype=np.int16)
            self.done = np.empty(self.memory_size, dtype=np.bool)

            # Minibatch memory
            self.states = np.empty((self.batch_size, self.stacked_frames, self.frame_height, self.frame_width),
                                   dtype=np.uint8)
            self.new_states = np.empty((self.batch_size, self.stacked_frames, self.frame_height, self.frame_width),
                                       dtype=np.uint8)
            self.indices = np.empty(self.batch_size, dtype=np.int32)

    # Adds a new state into the replay memory
    def add_experience(self, observation, action, reward, done):
        self.observations[self.current, ...] = observation
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.done[self.current] = done
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    # Returns a stack of frames according to the index given
    def get_observations(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.stacked_frames - 1:
            raise ValueError("Index must be at least {}!".format(self.stacked_frames - 1))
        return self.observations[index - self.stacked_frames + 1:index + 1, ...]

    # Pulls a number of indices from the replay memory according to the batch
    # size so that they are no duplicates, overlapping or terminal states
    def generate_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.stacked_frames, self.count - 1)
                if index >= self.current >= index - self.stacked_frames:
                    continue
                if self.done[index - self.stacked_frames:index].any():
                    continue
                break
            self.indices[i] = index

    # Returns a batch of valid states containing stacks of the current state and
    # incoming state, actions, rewards and whether the state terminated the environment
    def get_batch(self):
        if self.count < self.stacked_frames:
            raise ValueError('Not enough entries to get a minibatch!')

        self.generate_indices()

        for i, index in enumerate(self.indices):
            self.states[i] = self.get_observations(index - 1)
            self.new_states[i] = self.get_observations(index)

        return np.transpose(self.states, axes=(0, 2, 3, 1)), np.transpose(self.new_states, axes=(0, 2, 3, 1)), \
               self.actions[self.indices], self.rewards[self.indices], self.done[self.indices]

    # Saves all the replay memory on separate pickle files
    def save_memory(self):
        for filename in self.file_names:
            with open("output/" + filename + ".npy", "wb+") as file:
                np.save(file, getattr(self, filename), allow_pickle=False)
                self.sync(file)

    def sync(self, fh):
        fh.flush()
        fsync(fh.fileno())


class ActionGetter:
    """ Returns either a random action or the predicted action from the Q-network based on an annealing value """

    def __init__(self, env, e_start=1, e_end=0.1, replay_memory_start=20000, e_annealing_frames=1000000):
        self.env = env
        self.e_start = e_start
        self.e_end = e_end
        self.replay_memory_start = replay_memory_start
        self.e_annealing_frames = e_annealing_frames

        # Slopes and intercepts for exploration decrease
        self.slope = -(self.e_start - self.e_end) / self.e_annealing_frames
        self.intercept = self.e_start - self.slope * self.replay_memory_start

    def get_action(self, session, frame_number, dqn, state, evaluation=False):
        if evaluation:
            e = 0.0
        elif frame_number < self.replay_memory_start:
            e = self.e_start
        elif self.replay_memory_start <= frame_number < self.replay_memory_start + self.e_annealing_frames:
            e = self.slope * frame_number + self.intercept
        elif frame_number >= self.replay_memory_start + self.e_annealing_frames:
            e = self.e_end

        if np.random.rand(1) < e:
            return self.env.action_space.sample()
        return session.run(dqn.prediction, feed_dict={dqn.input: [state]})[0]


class PreprocessFrame:
    """ Resizes, crops and converts to RGB frames the Retro Environment """

    def __init__(self, frame_height=63, frame_width=113):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame = tf.placeholder(shape=[224, 256, 3], dtype=tf.uint8)

        self.processed = tf.image.rgb_to_grayscale(self.frame)
        self.processed = tf.image.crop_to_bounding_box(self.processed, 62, 15, 126, 226)
        self.processed = tf.image.resize_images(self.processed, [self.frame_height, self.frame_width],
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def process(self, session, frame):
        return session.run(self.processed, feed_dict={self.frame: frame})
