#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gym.envs.classic_control import rendering

class BaseObject(object):
    __env = None
    state = None

    @property
    def env(self):
        return self.__env

    @env.setter
    def env(self, e):
        self.__env = e


    def __init__(self, *args, **kwargs):
        for k in self.props:
            if k in kwargs:
                setattr(self, k, kwargs[k])
            else:
                setattr(self, k, getattr(self, 'default_%s'%k))


    def __setstate__(self, state):
        for k in self.props:
            if k in state:
                setattr(self, k, state[k])
            else:
                setattr(self, k, getattr(self, 'default_%s'%k))

    def create_shape(self):
        # create a shape to draw the object
        raise NotImplementedError

    def draw(self, viewer):
        self.create_shape()
        viewer.add_geom(self.shape)
        self.create_transform()


class Object(BaseObject):
    '''A simple object
    drawn as a circle by default
    '''

    props = ('name', 'coordinate', 'color', 'size', 'type')
    default_name = ''
    default_type = ''
    default_coordinate = (0,0)
    default_color = (0,0,0)
    default_size = 10


    def create_shape(self):
        self.shape = rendering.make_circle(self.size)
        self.shape.set_color(*self.color)

    def create_transform(self):
        if hasattr(self, 'coordinate') and self.coordinate:
            self.transform = rendering.Transform(translation=self.coordinate)
        else:
            self.transform = rendering.Transform()
        self.shape.add_attr(self.transform)


