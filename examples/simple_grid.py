#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Demo of RL V2.0

A robot playing the grid world, tries to find the gold, meanwhile
it has to avoid of the traps
"""

from skinner import *
from gym.envs.classic_control import rendering

from objects import _Object


class MyGridWorld(GridMaze, SingleAgentEnv):
    """Grid world
    
    A robot playing the grid world, tries to find the gold (yellow circle), meanwhile
    it has to avoid of the traps (orange and red circles)

    Extends:
        GridMaze, SingleAgentEnv
    
    Variables:
        metadata {dict} -- configuration of rendering
    """

    def config(self, conf):
        if isinstance(conf, str):
            import yaml
            with open(conf) as fo:
                s = fo.read()
            conf = yaml.unsafe_load(s)
        for k, v in conf.items():
            if isinstance(v, Object):
                setattr(self, k.upper(), v.position)
                self.add_objects({k:v})
            elif isinstance(v, ObjectGroup):
                setattr(self, k.upper(), [vi.position for vi in v])
                self.add_objects({k:v})
            else:
                if k != 'walls':
                    setattr(self, k, v)
                else:
                    self.add_walls(v)

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


class MyGridWorldx(MyGridWorld):
    def is_terminal(self):
        return self.agent.position in self.DEATHTRAPS or self.agent.position == self.GOLD or self.angent.power <= 0
