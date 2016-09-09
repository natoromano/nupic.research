# HTM Agent for Cartpole on OpenAI Gym

import sys

import gym
import numpy as np
import random

from htm_learner import HTMLearner



def run(env, num_trials=1000, num_steps=200):

  learner = HTMLearner(env)
  ave_cumu_r = None
  values = []

  for trial in range(num_trials):
    observation = env.reset()
    action = env.action_space.sample()
    state = learner.compute(observation, action)
    cum_rw = 0

    for t in range(num_steps):
      env.render()

      action = learner.bestAction(state)

      observation, reward, done, info = env.step(action)
      cum_rw += learner.discount * reward
      newState = learner.compute(observation, action)
      learner.update(state, action, newState, reward)
      state = newState

      if done:
        learner._reset()
        values.append(t + 1)
        k = 0.01
        if ave_cumu_r is None:
          ave_cumu_r = cum_rw
        else:
          ave_cumu_r = k * cum_rw + (1 - k) * ave_cumu_r
        if cum_rw > ave_cumu_r:
          learner.epsilon *= learner.eps_decay
        print "Number of steps at trial {} is {}".format(trial, t + 1)
        print "Average of last 100 is {:.2f}".format(np.mean(values[-100:]))
        print
        break



if __name__ == "__main__":
  random.seed(42)
  np.random.seed(42)

  env = gym.make('CartPole-v0')
  run(env)
