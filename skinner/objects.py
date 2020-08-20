#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gym.envs.classic_control import rendering

class BaseObject(object):
    props = ('name', 'coordinate', 'color', 'size', 'type')
    default_name = ''
    default_type = ''
    default_coordinate = (0,0)
    default_color = (0,0,0)
    default_size = 10

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


class Object(BaseObject):
    '''[Summary for Class Object]Object has 1 (principal) proptery
    state: state'''
    def __init__(self, state=None, *args, **kwargs):
        super(Object, self).__init__(*args, **kwargs)
        self.state = state

    def draw(self, viewer):
        self.shape = rendering.make_circle(self.size)
        if hasattr(self, 'coordinate') and self.coordinate:
            self.transform = rendering.Transform(translation=self.coordinate)
        else:
            self.transform = rendering.Transform()
        self.shape.add_attr(self.transform)
        self.shape.set_color(*self.color)
        viewer.add_geom(self.shape)


