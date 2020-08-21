#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .envs import *
from .objects import Object

class GridWorld(SingleAgentEnv):
    """Grid world
    
    Extends:
        gym.Env
    
    Variables:
        metadata {dict} -- configuration of rendering
    """

    n_cols = n_rows = 10
    edge = 20
    offset = 0.5

    @classmethod
    def t(cls, k, l, offset=0):
        return (k+offset)*cls.edge, (l+offset)*cls.edge


    def draw_background(self):
        # background of the grid world
        offset = 0.5
        for k in range(self.n_cols+1):
            line = rendering.Line(self.t(k, 0, offset), self.t(k, self.n_rows, offset))
            line.set_color(0,0,0)
            self.viewer.add_geom(line)
        for k in range(self.n_rows+1):
            line = rendering.Line(self.t(0, k, offset), self.t(self.n_cols, k, offset))
            line.set_color(0,0,0)
            self.viewer.add_geom(line)

    def create_viewer(self):
        screen_width, screen_height = self.t(self.n_cols+1, self.n_rows+1)
        self.viewer = rendering.Viewer(screen_width, screen_height)

    def render(self, mode='human', close=False):
        super(GridWorld, self).render(mode, close)
        self.agent.position = self.agent.state
        self.agent.draw(self.viewer, flag=False)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

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

def _coordinate(position, edge=10, offset=0.5):
    return (position[0]+0.5-offset)*edge, (position[1]+0.5-offset)*edge

class GridMaze(GridWorld):
    """Grid world with walls or barriers
    
    Extends:
        GridWorld
    """
    __walls = set()

    def add_walls(self, walls):
        self.__walls |= walls

    @property
    def walls(self):
        return self.__walls

    def draw_walls(self):

        class Wall(Object):
            '''Wall Class
            Black rectangles in the env
            '''

            props = ('name', 'position', 'color', 'size')
            default_color = (0, 0, 0)
            default_position=(0, 0)

            @property
            def coordinate(obj):
                return _coordinate(obj.position, edge=self.edge, offset=self.offset)

            def create_shape(obj):
                a = self.edge *.95 / 2
                obj.shape = rendering.make_polygon([(-a,-a),(a,-a),(a,a),(-a,a)])
                obj.shape.set_color(*obj.color)

        for wall in self.walls:
            wall = Wall(position=wall)
            wall.draw(self.viewer)

    def draw_background(self):
        # background of the grid world
        super(GridMaze, self).draw_background()
        self.draw_walls()

