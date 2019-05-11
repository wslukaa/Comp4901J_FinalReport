import numpy as np
import gym
import matplotlib.pyplot as plt
import csv

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy


ENV_NAME = 'CartPole-v0'

env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

policy = BoltzmannQPolicy()
sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

train_history=sarsa.fit(env, nb_steps=50000, visualize=False, verbose=2)

train_reward=train_history.history['episode_reward']

with open('sarsa_data.csv','w')as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(train_reward)

sarsa.save_weights('sarsa_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

train_reward=train_history.history['episode_reward']
plt.plot(train_reward)
plt.savefig('sarsa_cartpole')
sarsa.test(env, nb_episodes=5, visualize=True)
