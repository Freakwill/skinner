#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Demo of RL V2.0

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


class MyGridWorld(GridMaze, SingleAgentEnv):
    """Grid world 格子世界
    
    A robot playing the grid world, tries to find the golden (yellow circle), meanwhile
    it has to avoid of the traps(black circles)
    在这个格子世界的一个机器人要去寻找金子（黄色圆圈），同时需要避免很多陷阱（黑色圆圈）
    
    Extends:
        gym.Env
    
    Variables:
        metadata {dict} -- configuration of rendering
    """

    n_cols = conf['n_cols']
    n_rows = conf['n_rows']
    edge = conf['edge']

    TRAPS = [trap.position for trap in traps]
    DEATHTRAPS = [trap.position for trap in deathtraps]
    GOLD = gold.position

    def __init__(self, *args, **kwargs):
        super(MyGridWorld, self).__init__(*args, **kwargs)
        self.add_walls(conf['walls'])

    def is_terminal(self):
        return self.state in self.DEATHTRAPS or self.state == self.GOLD

    def is_successful(self):
        return self.state == self.GOLD

    def draw_objects(self):
        for trap in traps:
            trap.draw(self.viewer)
        # deathtraps
        for trap in deathtraps:
            trap.draw(self.viewer)
        # gold
        gold.draw(self.viewer)

        # robot
        self.agent.draw(self.viewer)

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

