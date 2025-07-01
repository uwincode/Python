#!/usr/bin/python3

"""
Run the model in training or testing mode

"""

import logging
import random

import gymnasium as gym
from tqdm import trange
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy.io as sio
import os


from src.common_definitions import TOTAL_EPISODES, UNBALANCE_P
from src.model import Brain
from src.utils import Tensorboard

from src.cacc import ACCEnv, CACCEnv


def main():  # pylint: disable=too-many-locals, too-many-statements
    """
    We create an environment, create a brain,
    create a Tensorboard, load weights, create metrics,
    create lists to store rewards, and then we run the training loop
    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    env_name = 'CACC'
    prediction_method = 'data' # 'constant', 'idm', or 'data' for CACC only

    render_env = False # 'Render the environment to be visually visible
    train = True # Train the network on the modified DDPG algorithm
    use_noise = True # OU Noise will be applied to the policy action
    eps_greedy = 0.95 # The epsilon for Epsilon-greedy in the policy's action
    warm_up = 1 # Following recommendation from OpenAI Spinning Up, the actions in the early epochs can be set random to increase exploration. This warm up defines how many epochs are initially set to do this
    tf_log_dir = './logs/DDPG/' # Save the logs of the training step

    # load training data
    train_data = sio.loadmat('trainSet.mat')['calibrationData']
    test_data = sio.loadmat('testSet.mat')['validationData']
    train_len = train_data.shape[0]
    test_len = test_data.shape[0]
    print('Number of training samples:', train_len)
    print('Number of validate samples:', test_len)

    if env_name == 'ACC':
        checkpoints_path = 'checkpoints/DDPG_'+ env_name + '_' # Save the weight of the network in the defined checkpoint file directory
        env = ACCEnv()
    elif env_name == 'CACC':
        checkpoints_path = 'checkpoints/DDPG_'+ env_name + '_' + prediction_method + '_' # Save the weight of the network in the defined checkpoint file directory
        env = CACCEnv(prediction_method)

    action_space_high = env.action_space.high[0]
    action_space_low = env.action_space.low[0]

    brain = Brain(env.observation_space.shape[0], env.action_space.shape[0], action_space_high,
                  action_space_low)
    tensorboard = Tensorboard(log_dir=tf_log_dir)

    # load weights if available
    # logging.info("Loading weights from %s*, make sure the folder exists", checkpoints_path)
    # brain.load_weights(checkpoints_path)

    # all the metrics
    acc_reward = tf.keras.metrics.Sum('reward', dtype=tf.float32)
    actions_squared = tf.keras.metrics.Mean('actions', dtype=tf.float32)
    Q_loss = tf.keras.metrics.Mean('Q_loss', dtype=tf.float32)
    A_loss = tf.keras.metrics.Mean('A_loss', dtype=tf.float32)

    # To store reward history of each episode
    ep_reward_list = []
    # To store collision count
    ep_collision_count = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    # run iteration
    with trange(TOTAL_EPISODES) as t:
        for ep in t:
            car_fol_id = random.randint(0, train_len - 1)
            data = train_data[car_fol_id, 0]
            prev_state, _ = env.reset(data)
            acc_reward.reset_state()
            actions_squared.reset_state()
            Q_loss.reset_state()
            A_loss.reset_state()
            brain.noise.reset()

            while True: # change
                if render_env:  # render the environment into GUI
                    env.render()

                # Receive state and reward from environment.
                cur_act = brain.act(
                    tf.expand_dims(prev_state, 0),
                    _notrandom=(
                        (ep >= warm_up)
                        and
                        (
                            random.random()
                            <
                            eps_greedy+(1-eps_greedy)*ep/TOTAL_EPISODES
                        )
                    ),
                    noise=use_noise
                )
                state, reward, done,truncated, info = env.step(cur_act)
                brain.remember(prev_state, reward, state, int(done))

                # Update weights
                if train:
                    c, a = brain.learn(brain.buffer.get_batch(unbalance_p=UNBALANCE_P))
                    Q_loss(c)
                    A_loss(a)

                # Post update for next step
                acc_reward(reward)
                actions_squared(np.square(cur_act/action_space_high))
                prev_state = state

                if done or truncated:
                    break

            ep_reward_list.append(acc_reward.result().numpy())
            ep_collision_count.append(env.collision_count)

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            avg_reward_list.append(avg_reward)

            # Print the average reward
            t.set_postfix(r=avg_reward)
            tensorboard(ep, acc_reward, actions_squared, Q_loss, A_loss)

            # Save weights
            if train and ep % 5 == 0:
                brain.save_weights(checkpoints_path)

    if train:
        brain.save_weights(checkpoints_path)

    logging.info("Training done...")

    # Plotting graph
    # Episodes versus Avg. Rewards
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    plt.figure()
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")

    plt.figure()
    plt.plot(ep_collision_count)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")

    plt.show()
    print("Graph plotted")


if __name__ == "__main__":
     main()
