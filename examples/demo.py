#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gym
import numpy as np

gym.register(
    id='GridWorld-v1',
    entry_point='simple_grid:MyGridWorld',
    max_episode_steps=200,
    reward_threshold=100.0
    )

from gym.spaces import Discrete

from skinner import *
from objects import Robot

import yaml
with open('config.yaml') as fo:
    s = fo.read()

conf = yaml.unsafe_load(s)
globals().update(conf)
WALLS = [wall.position for wall in walls]


class MyRobot(Robot):
    action_index = Discrete(4)
    actions = ['n','e','s','w']

    alpha = 0.3
    gamma = 0.9

    size = 30
    color = (0.8, 0.6, 0.4)

    def _next_state(self, state, action, env=None):
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

    def _get_reward(self, last_state, action, state, env=None):
        return env._get_reward(last_state, action, state)

    def reset(self):
        super(MyRobot, self).reset()
        self.state = 1, N

    # @property
    # def coordinate(self):
    #     return _coordinate(self.position)

agent = MyRobot({})


if __name__ == '__main__':
    env = gym.make('GridWorld-v1', agent=agent)
    env.seed()
    env.demo()
