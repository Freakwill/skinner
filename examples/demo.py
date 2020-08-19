#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gym
import numpy as np

gym.register(
    id='GridWorld-v0',
    entry_point='grid_mdp:GridWorld',
    max_episode_steps=200,
    reward_threshold=100.0
    )

from gym.spaces import Discrete

from skinner import *

import yaml
with open('config.yaml') as fo:
    s = fo.read()

conf = yaml.unsafe_load(s)
globals().update(conf)
WALLS = [wall.position for wall in walls]

class MyAgent(StandardAgent):
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
        last_state = state
        if action=='e':
            if state[0]<=M-1:
                state = (state[0]+1, state[1])
        elif action=='w':
            if state[0]>=2:
                state = (state[0]-1, state[1])
        elif action=='s':
            if state[1]>=2:
                state = (state[0], state[1]-1)
        elif action=='n':
            if state[1]<=N-1:
                state = (state[0], state[1]+1)
        else:
            raise Exception('invalid action!')
        if state in WALLS:
            state = last_state
        return state

    def _get_reward(self, env, last_state, action, state):
        return env._get_reward(last_state, action, state)

    def reset(self):
        super(MyAgent, self).reset()
        self.state = 1, N

agent = MyAgent({})

if __name__ == '__main__':
    env = gym.make('GridWorld-v0', agent=agent)
    env.seed()
    env.demo()
