#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from skinner import *

edge = 80
def _coordinate(position, offset=0.5):
    return (position[0]-offset)*edge+edge//2, (position[1]-offset)*edge+edge//2


class _Object(Object):
    props = ('name', 'position', 'color', 'size')
    default_position=(0,0)

    # def __init__(self, name='', state=None, position=(1,1), color=(0,0,0), size=30):
    #     self.position = position
    #     super(_Object, self).__init__(state, name, coordinate=None, color=color, size=size)

    @property
    def coordinate(self):
        return _coordinate(self.position)
    


class Trap(_Object):
    '''[Summary for Class Trap]'''
    pass
        
class DeathTrap(Trap):
    '''[Summary for Class Trap]'''
    def __init__(self, name='', state=None, position=(1,1), color=(1,0,0), size=30):
        super(DeathTrap, self).__init__(state, name, position=position, color=color, size=size)
        

class Gold(_Object):
    def draw(self, viewer):
        super(Gold, self).draw(viewer)
        gold_hole = _Object(name='gold_hole', state=True, position=self.position, color=(1, 1, 1), size=15)
        gold_hole.draw(viewer)


class Robot(_Object, StandardAgent):
    pass
    # def create_shape(self):
    #     from gym.envs.classic_control import rendering
    #     self.shape = rendering.Image('robot.jpeg', width=30, height=30)

class NeuralRobot(_Object, NeuralAgent):

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
