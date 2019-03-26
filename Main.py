import os
import tensorflow as tf
import numpy as np
from Utilities import ReplayMemory, ActionGetter, generate_gif
from MortalKombat import Environment
from DDQN import Keras_DDQN, DDQN, TargetNetworkUpdater
from Hyperparameters import *

os.makedirs(SAVER, exist_ok=True)
os.makedirs(SUMMARIES, exist_ok=True)
os.makedirs(SAVER, exist_ok=True)
os.makedirs(GIF, exist_ok=True)
SUMM_WRITER = tf.summary.FileWriter(SUMMARIES)

tf.reset_default_graph()

snes = Environment(GAME, STATE, FRAME_HEIGHT, FRAME_WIDTH)

with tf.variable_scope('mainDQN'):
    MAIN_DQN = Keras_DDQN(n_actions=snes.env.action_space.n)
with tf.variable_scope('targetDQN'):
    TARGET_DQN = Keras_DDQN(n_actions=snes.env.action_space.n)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')

LAYER_IDS = ["conv1", "conv2", "conv3", "conv4", "denseAdvantage",
             "denseAdvantageBias", "denseValue", "denseValueBias"]

# Loss, reward and evaluation score
with tf.name_scope('Performance'):
    LOSS_PH = tf.placeholder(tf.float32, shape=None, name='loss_summary')
    LOSS_SUMMARY = tf.summary.scalar('loss', LOSS_PH)
    REWARD_PH = tf.placeholder(tf.float32, shape=None, name='reward_summary')
    REWARD_SUMMARY = tf.summary.scalar('reward', REWARD_PH)
    EVAL_SCORE_PH = tf.placeholder(tf.float32, shape=None, name='evaluation_summary')
    EVAL_SCORE_SUMMARY = tf.summary.scalar('evaluation_score', EVAL_SCORE_PH)

PERFORMANCE_SUMMARIES = tf.summary.merge([LOSS_SUMMARY, REWARD_SUMMARY])

# Histogram summaries
with tf.name_scope('Parameters'):
    ALL_PARAM_SUMMARIES = []
    for i, Id in enumerate(LAYER_IDS):
        with tf.name_scope('mainDQN/'):
            MAIN_DQN_KERNEL = tf.summary.histogram(Id, tf.reshape(MAIN_DQN_VARS[i], shape=[-1]))
        ALL_PARAM_SUMMARIES.extend([MAIN_DQN_KERNEL])
PARAM_SUMMARIES = tf.summary.merge(ALL_PARAM_SUMMARIES)


def learn(session, replay_memory, main_dqn, target_dqn, batch_size, gamma):
    states, actions, rewards, new_states, terminal_flags = replay_memory.get_minibatch()

    arg_q_max = session.run(main_dqn.prediction, feed_dict={main_dqn.input: new_states})

    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input: new_states})
    double_q = q_vals[range(batch_size), arg_q_max]

    target_q = rewards + (gamma * double_q * (1 - terminal_flags))
    loss, _ = session.run([main_dqn.loss, main_dqn.update],
                          feed_dict={main_dqn.input: states,
                                     main_dqn.target_q: target_q,
                                     main_dqn.action: actions})
    return loss


def train():
    my_replay_memory = ReplayMemory(FRAME_HEIGHT, FRAME_WIDTH, BATCH_SIZE, STACKED_FRAMES, MEMORY_SIZE)
    network_updater = TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
    action_getter = ActionGetter(snes.env.action_space.n, replay_memory_start_size=REPLAY_MEMORY_START_SIZE)

    with tf.Session() as sess:
        if LOAD_MODEL:
            saver.restore(sess, SAVER + SAVED_FILE_NAME)
            print("Model restored.")
            with open("last_frame.dat") as file:
                frame_number = int(file.readline())
        else:
            sess.run(init)
            SUMM_WRITER.add_graph(sess.graph)
            frame_number = 0
        beginning_frame = frame_number
        rewards = []
        loss_list = []

        while frame_number < MAX_FRAMES:
            epoch_frame = 0

            while epoch_frame < EPOCH_FRAMES:
                snes.reset(sess, STACKED_FRAMES)
                reward_sum = 0

                for _ in range(MAX_EPISODE_LENGTH):
                    action = action_getter.get_action(sess, frame_number, snes.state, MAIN_DQN)
                    observation, reward, done, _ = snes.step(sess, action)
                    frame_number += 1
                    epoch_frame += 1
                    reward_sum += reward

                    my_replay_memory.add_experience(action=action, observation=observation, reward=reward, done=done)

                    if frame_number % STACKED_FRAMES == 0 and frame_number > REPLAY_MEMORY_START_SIZE + beginning_frame:
                        loss = learn(sess, my_replay_memory, MAIN_DQN, TARGET_DQN, BATCH_SIZE, GAMMA)
                        loss_list.append(loss)
                    if frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE + beginning_frame:
                        network_updater.update_networks(sess)
                    if done:
                        break

                rewards.append(reward_sum)

                # Output the progress:
                if len(rewards) % 10 == 0:
                    # Scalar summaries for tensorboard
                    if frame_number > REPLAY_MEMORY_START_SIZE + beginning_frame:
                        summ = sess.run(PERFORMANCE_SUMMARIES,
                                        feed_dict={LOSS_PH: np.mean(loss_list),
                                                   REWARD_PH: np.mean(rewards[-100:])})

                        SUMM_WRITER.add_summary(summ, frame_number)
                        loss_list = []
                    # Histogramm summaries for tensorboard
                    summ_param = sess.run(PARAM_SUMMARIES)
                    SUMM_WRITER.add_summary(summ_param, frame_number)

                    print(len(rewards), frame_number, np.mean(rewards[-100:]))
                    with open('rewards.dat', 'a') as reward_file:
                        print(len(rewards), frame_number, np.mean(rewards[-100:]), file=reward_file)

            print("Evaluating...")
            done = True
            gif = True
            episodes = 0
            reward_sum = 0
            frames_for_gif = []
            eval_rewards = []

            while episodes < EVAL_REPLAYS:
                if done:
                    snes.reset(sess, STACKED_FRAMES)
                    reward_sum = 0

                action = action_getter.get_action(sess, frame_number, snes.state, MAIN_DQN, evaluation=True)
                processed_observation, reward, done, observation = snes.step(sess, action)
                reward_sum += reward

                if gif:
                    frames_for_gif.append(observation)
                if done:
                    episodes += 1
                    eval_rewards.append(reward_sum)
                    gif = False

            print("Average score in {} replays: {}".format(EVAL_REPLAYS, np.mean(eval_rewards)))
            try:
                print("Making gif...")
                generate_gif(frame_number, frames_for_gif, eval_rewards[0], GIF)
            except IndexError:
                print("No evaluation game finished")

            # Save the network parameters
            save_path = saver.save(sess, SAVER + SAVED_FILE_NAME)
            print("Model saved in path: {}".format(save_path))

            with open('last_frame.dat', 'a') as file:
                file.truncate(0)
                print(frame_number, file=file)

            # Show the evaluation score in tensorboard
            summ = sess.run(EVAL_SCORE_SUMMARY, feed_dict={EVAL_SCORE_PH: np.mean(eval_rewards)})
            SUMM_WRITER.add_summary(summ, frame_number)
            with open('rewardsEval.dat', 'a') as eval_reward_file:
                print(frame_number, np.mean(eval_rewards), file=eval_reward_file)


def test():
    action_getter = ActionGetter(snes.env.action_space.n, replay_memory_start_size=REPLAY_MEMORY_START_SIZE)

    with tf.Session() as sess:
        saver.restore(sess, SAVER + SAVED_FILE_NAME)
        print("Model restored.")

        frames_for_gif = []
        snes.reset(sess, STACKED_FRAMES)
        reward_sum = 0
        while True:
            snes.env.render()
            action = action_getter.get_action(sess, 0, snes.state, MAIN_DQN, evaluation=True)
            processed_new_frame, reward, done, new_frame = snes.step(sess, action)
            reward_sum += reward
            frames_for_gif.append(new_frame)
            if done:
                break

        snes.env.close()
        print("Reward: {}".format(reward_sum))
        print("Making gif...")
        generate_gif(0, frames_for_gif, reward_sum, GIF)


if TRAIN:
    train()
else:
    test()
