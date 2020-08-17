#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gym.envs.classic_control import rendering

class BaseObject(object):

    def __init__(self, name, coordinate, color, size):
        self.name = name
        self.coordinate = coordinate
        self.color = color
        self.size = size


    def __setstate__(self, state):
        self.name, self.coordinate, self.color, self.size = state['name'], state['coordinate'], state['color'], state['size']


class Object(BaseObject):
    '''[Summary for Class Object]Object has 1 (principal) proptery
    state: state'''
    def __init__(self, state=None, *args, **kwargs):
        super(Object, self).__init__(*args, **kwargs)
        self.state = state

    def draw(self, viewer):
        if self.state is None:
            return None
        shape = rendering.make_circle(self.size)
        shape.add_attr(rendering.Transform(translation=self.coordinate))
        shape.set_color(*self.color)
        viewer.add_geom(shape)
