#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from skinner import *

class _Object(Object):
    # Mixin class for objects
    props = ('name', 'position', 'color', 'proportion')
    default_position = (0, 0)
    default_proportion = 0.5

    @property
    def size(self):
        return self.proportion * self.env.edge


class Trap(_Object):
    '''Common Trap'''
    pass
        
class DeathTrap(Trap):
    '''Death Trap
    '''
    default_color = (1,0,0)

    def reset(self):
        if not hasattr(self, 'shape'):
            self.create_shape()
        self.shape.set_color(*self.color)

    # def step(self):
    #     if self.button.pressed:
    #         self.shape.set_color(0, 0.8, 0)

class Gold(_Object):
    def draw(self, viewer):
        super(Gold, self).draw(viewer)
        r = self.env.edge / 7
        shape = rendering.make_circle(r)
        shape.set_color(1,1,1)
        viewer.add_geom(shape)
        shape.add_attr(self.transform)

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
        logo = rendering.Compound([shape1, shape2])
        logo.set_color(1,0.9,1)
        viewer.add_geom(logo)
        logo.add_attr(self.transform)


class Button(_Object):
    default_size = 20

from skinner import FiniteSet

class Robot(_Object, StandardAgent):
    action_space = FiniteSet('news')
    color = (0, 0.4, .7)

    def create_shape(self):
        length, width = self.env.edge / 4, self.env.edge / 2
        self.shape = rendering.make_capsule(length, width)
        self.shape.set_color(*self.color)
        r = self.env.edge / 10
        left_eye = rendering.make_circle(r)
        left_eye.add_attr(rendering.Transform(translation=(-r, 0)))
        right_eye = rendering.make_circle(r)
        right_eye.add_attr(rendering.Transform(translation=(2*r, 0)))
        self.eyes = rendering.Compound([left_eye, right_eye])
        self.eyes.set_color(.7,0.2,0.2)

    def draw(self, viewer, flag=True):
        if flag:
            super(Robot, self).draw(viewer, flag=True)
            viewer.add_geom(self.eyes)
            self.eyes.add_attr(self.transform)
        else:
            self.transform.set_translation(*self.coordinate)

    @property
    def position(self):
        return self.state[:2]

class SarsaRobot(Robot, SarsaAgent):
    pass

class SmartRobot(Robot):
    pass

class BoltzmannRobot(Robot, BoltzmannAgent):
    pass

from skinner.policies import *
class BayesRobot(BoltzmannRobot):
    '''BayesRobot
    Agent with action seletion method based on naive Bayes
    '''
    def __init__(self, *args, **kwargs):
        super(BayesRobot, self).__init__(*args, **kwargs)
        self.state_count = {self.init_state:1}
        self.action_count ={a:0 for a in self.action_space}

    def update(self, action, reward):
        super(BayesRobot, self).update(action, reward)
        if self.state in self.state_count:
            self.state_count[self.state] += 1
        else:
            self.state_count[self.state] = 1
        if action in self.action_count:
            self.action_count[action] += 1
        else:
            self.action_count[action] = 1

    @property
    def state_proba(self):
        C = self.state_count
        N = np.sum([n for s, n in C.items()])
        return {s: n/N for s, n in C.items()}

    @property
    def action_proba(self):
        C = self.action_count
        N = np.sum([n for s, n in C.items()])
        l = 0.001
        return {a: (C[a]+l)/(N+l*self.action_space.n) for a in self.action_space}


    def select_action(self):
        if len(self.QTable)<10:
            return super(BayesRobot, self).select_action()
        return bayes(self.state, self.action_space, self.QTable, epsilon=self.epsilon, temperature=self.temperature, pa=self.action_proba, ps=self.state_proba)



from sklearn.preprocessing import OneHotEncoder
_encoder = OneHotEncoder()

class NeuralRobot(Robot, NeuralAgent):

    @classmethod
    def key2vector(cls, key):
        return *key[0], *cls.action_space.encode(k=key[1], encoder=_encoder)


    def get_samples(self, size=0.8):
        # state, action, reward, next_state ~ self.cache
        L = len(self.cache)
        size = int(size * L)
        inds = np.random.choice(L, size=size)
        states = self.cache.loc[inds, 'state'].values
        states = np.array([state for state in states])
        actions = self.cache.loc[inds, 'action'].values
        actions = self.action_space.encode(k=actions, encoder=_encoder)
        rewards = self.cache.loc[inds, 'reward'].values
        next_states = self.cache.loc[inds, 'state+'].values
        X = np.column_stack((states, actions))
        y = rewards + self.gamma * np.array([self.V_(s) for s in next_states])
        return X, y
