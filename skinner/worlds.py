#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .envs import *
from .objects import Object

class GridWorld(BaseEnv):
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

    def add_objects(self, objs):
        super(GridWorld, self).add_objects(objs)
        import types
        def _coordinate(o):
            return self.coordinate(o.position)
        for obj in objs:
            obj._coordinate = types.MethodType(_coordinate, obj)


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
        self.agent.draw(self.viewer, flag=False)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def collide(self, x):
        return not (1<=x[0]<=self.n_rows and 1<=x[1]<=self.n_cols)


def _coordinate(position, edge=10, offset=0.5):
    return (position[0]+0.5-offset)*edge, (position[1]+0.5-offset)*edge


class Wall(Object):
    '''Wall Class
    Black rectangles in the env
    '''
    __env = None

    props = ('name', 'position', 'color', 'size')
    default_color = (0, 0, 0)
    default_position=(0, 0)

    @property
    def coordinate(self):
        return _coordinate(self.position, edge=self.env.edge, offset=self.env.offset)

    def create_shape(self):
        a = self.env.edge *.95 / 2
        self.shape = rendering.make_polygon([(-a,-a), (a,-a), (a,a), (-a,a)])
        self.shape.set_color(*self.color)


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

        for wall in self.walls:
            wall = Wall(position=wall)
            wall.env = self
            wall.draw(self.viewer)

    def draw_background(self):
        # background of the grid world
        super(GridMaze, self).draw_background()
        self.draw_walls()


    def collide(self, x):
        return super(GridMaze, self).collide(x) or any([wall == x for wall in self.walls])

