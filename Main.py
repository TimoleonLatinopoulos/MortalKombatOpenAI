import os
import tensorflow as tf
import numpy as np
from Utilities import ReplayMemory, ActionGetter, generate_gif
from MortalKombat import Environment
from DDQN import DDQN, TargetNetworkUpdater
from Hyperparameters import *

os.makedirs(SAVER, exist_ok=True)
os.makedirs(SUMMARIES, exist_ok=True)
os.makedirs(SAVER, exist_ok=True)
os.makedirs(GIF, exist_ok=True)
WRITER = tf.summary.FileWriter(SUMMARIES)

tf.reset_default_graph()

snes = Environment(GAME, STATE, FRAME_HEIGHT, FRAME_WIDTH)

with tf.variable_scope('mainDQN'):
    MAIN_DQN = DDQN(snes.env.action_space.n, FRAME_HEIGHT, FRAME_WIDTH, STACKED_FRAMES, LEARNING_RATE)
with tf.variable_scope('targetDQN'):
    TARGET_DQN = DDQN(snes.env.action_space.n, FRAME_HEIGHT, FRAME_WIDTH, STACKED_FRAMES, LEARNING_RATE)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')

# Loss, reward and evaluation score
with tf.name_scope('Performance'):
    LOSS_PH = tf.placeholder(tf.float32, shape=None, name='loss_summary')
    LOSS_SUMMARY = tf.summary.scalar('loss', LOSS_PH)
    REWARD_PH = tf.placeholder(tf.float32, shape=None, name='reward_summary')
    REWARD_SUMMARY = tf.summary.scalar('reward', REWARD_PH)
    EVAL_SCORE_PH = tf.placeholder(tf.float32, shape=None, name='evaluation_summary')
    EVAL_SCORE_SUMMARY = tf.summary.scalar('evaluation_score', EVAL_SCORE_PH)

PERFORMANCE_SUMMARIES = tf.summary.merge([LOSS_SUMMARY, REWARD_SUMMARY])


def learn(session, replay_memory, main_dqn, target_dqn, batch_size, gamma):
    states, new_states, actions, rewards, done = replay_memory.get_batch()

    arg_q_max = session.run(main_dqn.prediction, feed_dict={main_dqn.input: new_states})

    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input: new_states})
    double_q = q_vals[range(batch_size), arg_q_max]

    target_q = rewards + (gamma * double_q * (1 - done))
    loss, _ = session.run([main_dqn.loss, main_dqn.update],
                          feed_dict={main_dqn.input: states,
                                     main_dqn.target_q: target_q,
                                     main_dqn.action: actions})
    return loss


def train():
    my_replay_memory = ReplayMemory(FRAME_HEIGHT, FRAME_WIDTH, STACKED_FRAMES, BATCH_SIZE, MEMORY_SIZE)
    network_updater = TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
    action_getter = ActionGetter(snes.env.action_space.n, replay_memory_start=REPLAY_MEMORY_START)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        if LOAD_MODEL:
            saver.restore(sess, SAVER + SAVED_FILE_NAME)
            print("Model restored.")
            with open("last_frame.dat") as file:
                frame_number = int(file.readline())
                episodes = int(file.readline())
        else:
            sess.run(init)
            WRITER.add_graph(sess.graph)
            frame_number = 0
            episodes = 0
        beginning_episode = episodes
        rewards = []
        loss_list = []

        while episodes < MAX_EPISODES:
            epoch_episodes = 0

            while epoch_episodes < EPOCH_EPISODES:
                snes.reset(sess, STACKED_FRAMES)
                reward_sum = 0

                for _ in range(MAX_EPISODE_LENGTH):
                    action = action_getter.get_action(sess, frame_number, MAIN_DQN, snes.state)
                    observation, _, reward, done = snes.step(sess, action)
                    frame_number += 1
                    reward_sum += reward

                    my_replay_memory.add_experience(observation=observation, action=action, reward=reward, done=done)

                    if episodes > REPLAY_MEMORY_START + beginning_episode:
                        if frame_number % STACKED_FRAMES == 0:
                            loss = learn(sess, my_replay_memory, MAIN_DQN, TARGET_DQN, BATCH_SIZE, GAMMA)
                            loss_list.append(loss)
                        if frame_number % UPDATE_FRAMES == 0:
                            network_updater.update_networks(sess)
                    if done:
                        break

                epoch_episodes += 1
                episodes += 1
                rewards.append(reward_sum)

                # Output the progress
                if episodes % SAVE_STEP == 0:
                    # Scalar summaries for tensorboard
                    if episodes > REPLAY_MEMORY_START + beginning_episode:
                        summ = sess.run(PERFORMANCE_SUMMARIES,
                                        feed_dict={LOSS_PH: np.mean(loss_list), REWARD_PH: np.mean(rewards)})
                        WRITER.add_summary(summ, episodes)
                        loss_list = []

                    print(episodes, frame_number, np.mean(rewards))
                    with open('rewards.dat', 'a') as reward_file:
                        print(episodes, frame_number, np.mean(rewards), file=reward_file)
                    rewards = []

            print("Evaluating...")
            done = True
            gif = True
            eval_episodes = 0
            reward_sum = 0
            frames_for_gif = []
            eval_rewards = []

            while eval_episodes < EVAL_REPLAYS:
                if done:
                    snes.reset(sess, STACKED_FRAMES)
                    reward_sum = 0

                action = action_getter.get_action(sess, frame_number, MAIN_DQN, snes.state, evaluation=True)
                processed_observation, observation, reward, done = snes.step(sess, action, evaluation=True)
                reward_sum += reward

                if gif:
                    frames_for_gif.append(observation)
                if done:
                    eval_episodes += 1
                    eval_rewards.append(reward_sum)
                    gif = False

            print("Average score in {} replays: {}".format(EVAL_REPLAYS, np.mean(eval_rewards)))

            # Save the network parameters
            if np.mean(eval_rewards) > 0:
                save_path = saver.save(sess, SAVER + SAVED_FILE_NAME_SUCCESS)
            else:
                save_path = saver.save(sess, SAVER + SAVED_FILE_NAME)
            print("Model saved in path: {}".format(save_path))

            with open('last_frame.dat', 'a') as file:
                file.truncate(0)
                print(frame_number, episodes, sep='\n', file=file)

            try:
                print("Making gif...")
                generate_gif(episodes, eval_rewards[0], frames_for_gif, GIF)
                print("Gif generated in path: " + GIF)
            except IndexError:
                print("No evaluation game finished")

            # Show the evaluation score in tensorboard
            summ = sess.run(EVAL_SCORE_SUMMARY, feed_dict={EVAL_SCORE_PH: np.mean(eval_rewards)})
            WRITER.add_summary(summ, episodes)
            with open('rewardsEval.dat', 'a') as eval_reward_file:
                print(episodes, np.mean(eval_rewards), file=eval_reward_file)


def test():
    action_getter = ActionGetter(snes.env.action_space.n, replay_memory_start=REPLAY_MEMORY_START)

    with tf.Session() as sess:
        saver.restore(sess, SAVER + SAVED_FILE_NAME)
        print("Model restored.")

        frames_for_gif = []
        snes.reset(sess, STACKED_FRAMES)
        reward_sum = 0
        while True:
            snes.env.render()
            action = action_getter.get_action(sess, 0, MAIN_DQN, snes.state, evaluation=True)
            processed_new_frame, new_frame, reward, done = snes.step(sess, action, evaluation=True)
            reward_sum += reward
            frames_for_gif.append(new_frame)
            if done:
                break
        snes.env.close()

        print("Reward: {}".format(reward_sum))
        try:
            print("Making gif...")
            generate_gif(0, reward_sum, frames_for_gif, GIF)
            print("Gif generated in path: " + GIF)
        except IndexError:
            print("No evaluation game finished")


if TRAIN:
    train()
else:
    test()
