#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gym.envs.classic_control import rendering

class BaseObject:
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

    def create_transform(self):
        raise NotImplementedError

    def draw(self, viewer=None):
        if viewer is None:
            viewer = self.env.viewer
        if not hasattr(self, 'shape'):
            self.create_shape()
        viewer.add_geom(self.shape)
        self.create_transform()

    def reset(self):
        raise NotImplementedError


class Object(BaseObject):
    '''A simple object
    drawn as a circle by default
    '''

    props = ('name', 'coordinate', 'color', 'size', 'type')
    default_name = ''
    default_type = ''
    default_coordinate = (0, 0)
    default_color = (0, 0, 0)
    default_size = 10

    def reset(self):
        pass

    def create_shape(self):
        self.shape = rendering.make_circle(self.size)
        self.shape.set_color(*self.color)

    def create_transform(self):
        if hasattr(self, 'coordinate') and self.coordinate:
            self.transform = rendering.Transform(translation=self.coordinate)
        else:
            self.transform = rendering.Transform()
        self.shape.add_attr(self.transform)

    @property
    def coordinate(self):
        """
        If an object has no absolute coordinate and is plotted with a relative position
        according to the env.
        One should define a method translating position to coordinate.
        It is recommanded to define the method dynamically in envs, for example,

        import types
        def _coordinate(o):
            return self.coordinate(o.position)
        for _, obj in objs.items():
            if isinstance(obj, Object):
                obj._coordinate = types.MethodType(_coordinate, obj)
        """
        return self._coordinate()


class ObjectGroup(BaseObject):
    props = ('name', 'members')
    default_name = ''

    def __init__(self, name='', members=[]):
        self.name = name
        self.__members = members

    def __getitem__(self, k):
        return self.members[k]

    def reset(self):
        for m in self.members:
            m.reset()


    @property
    def members(self):
        return self.__members

    @members.setter
    def members(self, ms):
        self.__members = ms

    def add_members(self, ms):
        return self.__members.extend(ms)
    

    def draw(self, viewer=None):
        for m in self.members:
            m.draw(viewer)

    @property
    def env(self):
        return self.__env

    @env.setter
    def env(self, e):
        self.__env = e
        for m in self.members:
            m.env = e

