import random
import numpy as np
import tensorflow as tf
import imageio
from skimage.transform import resize


def generate_gif(number, frames, reward, path):
    for i, frame in enumerate(frames):
        frames[i] = frame
        # frames[i] = resize(frame, (448, 512, 3), preserve_range=True, order=0).astype(np.uint8)
    imageio.mimsave(f'{path}{"MortalKombat_frame{}_reward_{}.gif".format(number, reward)}',
                    frames, duration=1 / 60)


class ReplayMemory:

    def __init__(self, frame_height=82, frame_width=128, batch_size=32, stacked_frames=4, memory_size=1000000):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.batch_size = batch_size
        self.stacked_frames = stacked_frames
        self.memory_size = memory_size

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

    def add_experience(self, action, observation, reward, done):
        if observation.shape != (self.frame_height, self.frame_width):
            raise ValueError('Wrong frame size!')

        self.observations[self.current, ...] = observation
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.done[self.current] = done
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.stacked_frames - 1:
            raise ValueError("Index must be at least {}".format(self.stacked_frames - 1))
        return self.observations[index - self.stacked_frames + 1:index + 1, ...]

    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.stacked_frames, self.count - 1)
                if index >= self.current >= index - self.stacked_frames:
                    continue
                if self.done[index - self.stacked_frames:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        if self.count < self.stacked_frames:
            raise ValueError('Not enough memories to get minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)

        return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[
            self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.done[self.indices]


class ActionGetter:

    def __init__(self, n_actions, eps_initial=1, eps_final=0.1, replay_memory_start_size=50000,
                 eps_annealing_frames=1000000):
        self.n_actions = n_actions
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.replay_memory_start_size = replay_memory_start_size
        self.eps_annealing_frames = eps_annealing_frames

        # Slopes and intercepts for exploration decrease
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope * self.replay_memory_start_size

    def get_action(self, session, frame_number, state, main_dqn, evaluation=False):

        if evaluation:
            eps = 0.0
        elif frame_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif self.replay_memory_start_size <= frame_number < self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope * frame_number + self.intercept

        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)
        return session.run(main_dqn.prediction, feed_dict={main_dqn.input: [state]})[0]


class ProcessFrame:

    def __init__(self, frame_height=82, frame_width=128):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame = tf.placeholder(shape=[224, 256, 3], dtype=tf.uint8)

        self.processed = tf.image.rgb_to_grayscale(self.frame)
        self.processed = tf.image.crop_to_bounding_box(self.processed, 45, 0, 150, 256)
        self.processed = tf.image.resize_images(self.processed, [self.frame_height, self.frame_width],
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def process(self, session, frame):
        return session.run(self.processed, feed_dict={self.frame: frame})
