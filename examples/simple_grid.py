#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Demo of RL

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

    def is_terminal(self):
        return self.agent.position in self.DEATHTRAPS or self.agent.position == self.GOLD

    def is_successful(self):
        return self.agent.position == self.GOLD

    def post_process(self):
        if self.is_successful():
            n = self.agent.n_steps
        else:
            n = self.max_steps
        self.history = self.history.append({'n_steps': n, 'total rewards':self.agent.total_reward}, ignore_index=True)
        self.agent.post_process()

    def pre_process(self):
        import pandas as pd
        self.history = pd.DataFrame(columns=('n_steps', 'total rewards'))


class MyGridWorld1(MyGridWorld):
    def is_terminal(self):
        return self.agent.position in self.DEATHTRAPS or self.agent.position == self.GOLD or self.agent.power <= 0


class MyGridWorld2(MyGridWorld):

    def is_terminal(self):
        return ((self.agent.position == self.DEATHTRAP1 and self.agent.flag1 == 0) or 
        (self.agent.position == self.DEATHTRAP2 and self.agent.flag2 == 0) or 
        self.agent.position == self.DEATHTRAP3 or self.agent.position == self.GOLD)

    def render(self, mode='human', close=False):
        super(MyGridWorld2, self).render(mode, close)
        if self.agent.flag1:
            self.objects['deathtrap1'].shape.set_color(0, 0.8, 0)
        if self.agent.flag2:
            self.objects['deathtrap2'].shape.set_color(0, 0.8, 0)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


class MyGridWorld3(MyGridWorld):

    def is_terminal(self):
        return ((self.agent.position == self.DEATHTRAP1 and self.agent.flag1 == 0) or 
        (self.agent.position == self.DEATHTRAP2 and self.agent.flag2 == 0) or 
        (self.agent.position == self.DEATHTRAP3 and self.agent.flag3 == 0) or 
        self.agent.position == self.DEATHTRAP4 or self.power <=0 or
        self.agent.position == self.GOLD)

    def render(self, mode='human', close=False):
        super(MyGridWorld2, self).render(mode, close)
        if self.agent.flag1:
            self.objects['deathtrap1'].shape.set_color(0, 0.8, 0)
        if self.agent.flag2:
            self.objects['deathtrap2'].shape.set_color(0, 0.8, 0)
        if self.agent.flag3:
            self.objects['deathtrap3'].shape.set_color(0, 0.8, 0)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


