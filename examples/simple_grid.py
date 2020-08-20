#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""强化学习演示程序V2.0

在这个格子世界的一个机器人要去寻找金子（黄色圆圈），同时需要避免很多陷阱（黑色圆圈）

主要方法:
step: 每次主体与环境互动的流程
    调用get_reward与nex_state
get_reward: 每次互动获得回报 r=g(s,a,s')
next_state: 转态迁移 s'=f(s,a)
"""

from skinner import *
from gym.envs.classic_control import rendering

from objects import *

import yaml
with open('config.yaml') as fo:
    s = fo.read()
conf = yaml.unsafe_load(s)
globals().update(conf)

screen_width = edge * (M+2)
screen_height = edge * (N+2)

TRAPS = [trap.position for trap in traps]
DEATHTRAPS = [trap.position for trap in deathtraps]
GOLD = gold.position
WALLS = [wall.position for wall in walls]

class MyGridWorld(GridWorld, SingleAgentEnv):
    """Grid world 格子世界
    
    A robot playing the grid world, tries to find the golden (yellow circle), meanwhile
    it has to avoid of the traps(black circles)
    在这个格子世界的一个机器人要去寻找金子（黄色圆圈），同时需要避免很多陷阱（黑色圆圈）
    
    Extends:
        gym.Env
    
    Variables:
        metadata {dict} -- configuration of rendering
    """

    M = conf['M']
    N = conf['N']
    edge = conf['edge']


    def _get_reward(self, state, action, next_state, env=None):
        """回报函数
        
        被step方法调用
        
        Arguments:
            state -- 动作之前的状态
            action -- 动作
            next_state -- 动作之后的状态
        
        Returns:
            number -- 回报值
        """
        if next_state in TRAPS:
            return -1
        elif next_state in DEATHTRAPS:
            return -2
        elif next_state == GOLD:
            return 3
        elif state == next_state:
            return -0.2
        else:
            return -0.05

    def is_terminal(self):
        return self.state in DEATHTRAPS or self.state == GOLD

    def is_successful(self):
        return self.state == GOLD

    def post_process(self):
        if self.is_successful():
            self.history['n_steps'].append(self.agent.n_steps)
        else:
            self.history['n_steps'].append(self.max_steps)
        self.history['reward'].append(self.agent.total_reward)
        self.agent.post_process()

    def pre_process(self):
        self.history['n_steps'] = []
        self.history['reward'] = []

    def end_process(self):
        import pandas as pd
        data = pd.DataFrame(self.history)
        data.to_csv('history.csv')

    def draw_objects(self):
        for trap in traps:
            trap.draw(self.viewer)
        # deathtraps
        for trap in deathtraps:
            trap.draw(self.viewer)
        # deathtraps
        for wall in walls:
            wall.draw(self.viewer)
        # gold
        gold.draw(self.viewer)

        # robot
        self.agent.draw(self.viewer)


    def render(self, mode='human', close=False):
        super(MyGridWorld, self).render(mode, close)
        self.agent.position = self.agent.state
        self.agent.draw(self.viewer, flag=False)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
