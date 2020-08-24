#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from skinner import *

class _Object(Object):
    props = ('name', 'position', 'color', 'size')
    default_position = (0, 0)

    @property
    def coordinate(self):
        """
        Here one should define a method translating position to coordinate.
        Currently, it has been defined in the env.
        """
        return self._coordinate()


class Trap(_Object):
    '''[Summary for Class Trap]'''
    pass
        
class DeathTrap(Trap):
    '''[Summary for Class Trap]'''
    default_size = 30
    default_color = (1,0,0)
        

class Gold(_Object):
    def draw(self, viewer):
        super(Gold, self).draw(viewer)
        gold_hole = _Object(name='gold_hole', state=True, position=self.position, color=(1, 1, 1), size=15)
        gold_hole.draw(viewer)

class Charger(_Object):
    def create_shape(self):
        a = self.env.edge *.6 / 2
        self.shape = rendering.make_polygon([(-a,-a), (a,-a), (a,a), (-a,a)])
        self.shape.set_color(*self.color)

    def draw(self, viewer):
        super(Charger, self).draw(viewer)
        a = self.env.edge *.6 / 2
        shape1 = rendering.make_polygon([(-a/2, a/4), (0, a/4*3), (0, a/4), (a/2, -a/4), (0, -a/4)])
        shape2 = rendering.make_polygon([(0,-a/4), (0,-a/4*3), (a/2, -a/4)])
        shape = rendering.Compound([shape1, shape2])
        shape.set_color(1,0.9,1)
        viewer.add_geom(shape)
        shape.add_attr(self.transform)


class Robot(_Object, StandardAgent):

    def create_shape(self):
        self.shape = rendering.make_capsule(self.length, self.width)
        self.shape.set_color(*self.color)

    @property
    def position(self):
        return self.state[:2]

class SmartRobot(Robot, NonStandardAgent):
    pass

class NeuralRobot(Robot, NeuralAgent):

    @classmethod
    def key2vector(cls, key):
        return *key[0], *cls.action_space.onehot_encode(key[1])


    def get_samples(self, size=0.8):
        # state, action, reward, next_state ~ self.cache
        L = len(self.cache)
        size = int(size * L)
        inds = np.random.choice(L, size=size)
        states = self.cache.loc[inds, 'state'].values
        states = np.array([state for state in states])
        actions = self.cache.loc[inds, 'action'].values
        actions = self.action_space.onehot_encode(actions)
        rewards = self.cache.loc[inds, 'reward'].values
        next_states = self.cache.loc[inds, 'state+'].values
        X = np.column_stack((states, actions))
        y = rewards + self.gamma * np.array([self.V_(s) for s in next_states])
        return X, y
