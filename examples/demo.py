#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gym
import numpy as np
from config import *

gym.register(
    id='GridWorld-v0',
    entry_point='grid_mdp:GridWorld',
    max_episode_steps=200,
    reward_threshold=100.0
    )

from gym.spaces import Discrete

from skinner import *
class MyAgent(NeuralAgent):
    action_index = Discrete(4)
    actions = ['n','e','s','w']

    alpha = 0.3
    gamma = 0.9

    def _next_state(self, state, action):
        """状态迁移方法
        
        这是一个确定性迁移
        
        Arguments:
            state
            action
        
        Returns:
            new state
        
        Raises:
            Exception -- invalid action
        """

        if action=='e':
            if state[0]<=M-1:
                state = (state[0]+1, state[1])
            return state
        elif action=='w':
            if state[0]>=2:
                state = (state[0]-1, state[1])
            return state
        elif action=='s':
            if state[1]>=2:
                state = (state[0], state[1]-1)
            return state
        elif action=='n':
            if state[1]<=N-1:
                state = (state[0], state[1]+1)
            return state
        else:
            raise Exception('invalid action!')

    def _get_reward(self, env, last_state, action, state):
        return env._get_reward(last_state, action, state)

agent = MyAgent({})

if __name__ == '__main__':
    env = gym.make('GridWorld-v0', agent=agent)
    env.seed()
    env.demo()
