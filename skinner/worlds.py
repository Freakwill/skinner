#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .envs import *

class GridWorld(MyEnv):
    """Grid world
    
    Extends:
        gym.Env
    
    Variables:
        metadata {dict} -- configuration of rendering
    """

    M = N = 10
    edge = 20

    @classmethod
    def t(cls, k, l, offset=0):
        return (k+offset)*cls.edge, (l+offset)*cls.edge


    def draw_background(self):
        # background of the grid world
        offset = 0.5  
        for k in range(self.M+1):
            line = rendering.Line(self.t(k, 0, offset), self.t(k, self.N, offset))
            line.set_color(0,0,0)
            self.viewer.add_geom(line)
        for k in range(self.N+1):
            line = rendering.Line(self.t(0, k, offset), self.t(self.M, k, offset))
            line.set_color(0,0,0)
            self.viewer.add_geom(line)

    def create_viewer(self):
        screen_width, screen_height = self.t(self.M+1, self.N+1)
        self.viewer = rendering.Viewer(screen_width, screen_height)
