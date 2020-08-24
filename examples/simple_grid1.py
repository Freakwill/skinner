#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Demo of RL V2.0

在这个格子世界的一个机器人要去寻找金子（黄色圆圈），同时需要避免很多陷阱（黑色圆圈）

"""

from skinner import *
from gym.envs.classic_control import rendering

from objects import *

import yaml
with open('config1.yaml') as fo:
    s = fo.read()
conf = yaml.unsafe_load(s)
globals().update(conf)


class MyGridWorld(GridMaze, SingleAgentEnv):
    """Grid world
    
    A robot playing the grid world, tries to find the golden (yellow circle), meanwhile
    it has to avoid of the traps(black circles)

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


    @classmethod
    def coordinate(cls, position, offset=None):
        """Transform a position to a coordinate
        
        Arguments:
            position {tuple} -- the position of an object in the grid world
        
        Keyword Arguments:
            offset {tuple} -- margin of the grid world (default: {None})

        Return:
            tuple -- the coordinate where the object lies
        """

        if offset is None:
            offset = cls.offset
        return (position[0]-offset)*cls.edge+cls.edge//2, (position[1]-offset)*cls.edge+cls.edge//2


    def __init__(self, *args, **kwargs):
        super(MyGridWorld, self).__init__(*args, **kwargs)
        self.add_walls(conf['walls'])
        self.add_objects((*traps, *deathtraps, gold))

    def is_terminal(self):
        return self.agent.position in self.DEATHTRAPS or self.agent.position == self.GOLD

    def is_successful(self):
        return self.agent.position == self.GOLD

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
