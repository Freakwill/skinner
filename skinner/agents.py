#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .policies import *
from .objects import *

class BaseAgent(Object):
    '''[Summary for Class BaseAgent]BaseAgent has 4 (principal) propteries
    QTable: QTable
    state: state
    last_state: last state
    init_state: init state'''
    env = None
    n_steps = 0
    total_reward = 0

    def next_state(self, action):
        """
        self.__next_state: state transition method
        function: state, action -> state
        """
        self.n_steps +=1
        self.last_state = self.state
        self.state = self._next_state(self.last_state, action)

    def get_reward(self, action):
        r = self._get_reward(self.last_state, action, self.state)
        self.total_reward += r
        return r

    def Q(self, key):
        raise NotImplementedError

    def V(self, state):
        raise NotImplementedError

    def visited(self, key):
        raise NotImplementedError

    def predict(self, key):
        return 0

    def update(self, key):
        raise NotImplementedError

    def step(self, env):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def reset(self, env=None):
        self.n_steps = 0
        self.total_reward = 0

    def _next_state(self, state, action):
        """transition function
        s, a -> s'
        """
        raise NotImplementedError

    def _get_reward(state0, action, state1):
        """reward function
        s, a, s' -> r
        """
        raise NotImplementedError


class StandardAgent(BaseAgent):

    def __init__(self, QTable={}, last_state=None, init_state=None):
        self.QTable = QTable
        self.VTable = {}
        for key, value in QTable.items():
            state, action = key
            if state not in self.VTable:
                self.VTable[state] = self.V(state)
            elif self.VTable[state] < q:
                self.VTable[state] = q
        self.last_state = last_state
        self.init_state = init_state
        self.state = init_state
        self.epoch = 1
        self.gamma = 0.9
        self.alpha = 0.1
        self.epsilon = 0.05

    def select_action(self, default_action=None):
        if default_action is None:
            default_action = self.actions.sample()
        return greedy(self.state, self.actions, self.Q, self.epsilon, default_action)


    def step(self):
        action = self.select_action()
        self.next_state(action)
        reward = self.get_reward(action)
        self.update(action, reward)
        return reward


    def Q(self, key):
        return self.QTable.get(key, 0)

    def V(self, state=None):
        if state is None:
            state = self.state
        if state in self.VTable:
            return self.VTable[state]
        else:
            return max([self.Q(key=(state, a))for a in self.actions])

    def visited(self, key):
        return key in self.QTable

    def update(self, action, reward):
        key = self.last_state, action
        state = self.last_state
        if key in self.QTable:
            self.QTable[key] += self.alpha * (reward + self.gamma * self.V() - self.QTable[key])
        else:
            # resume self.QTable[key] == 0
            self.QTable[key] = self.alpha * reward + self.gamma * self.V()
        q = self.QTable[key]
        if state not in self.VTable:
            self.VTable[state] = self.V(state)
        elif self.VTable[state] < q:
            self.VTable[state] = q

    def draw(self, viewer, flag=True):
        if flag:
            super(StandardAgent, self).draw(viewer)
        else:
            self.transform.set_translation(*self.coordinate)

    def post_process(self, *args, **kwargs):
        self.epsilon **= .99
        self.alpha **= .99

import pandas as pd
from sklearn.neural_network import *

class NeuralAgent(StandardAgent):

    def __init__(self, state=None, last_state=None, init_state=None):
        self.state = state
        self.last_state = last_state
        self.init_state = init_state
        self.mainQ = MLPRegressor(hidden_layer_sizes=(10,), max_iter=20, warm_start=True)
        self.targetQ = MLPRegressor(hidden_layer_sizes=(10,))
        self.epoch = 1
        self.gamma = 0.9
        self.alpha = 0.1
        self.epsilon = 0.2
        self.cache = pd.DataFrame(columns=('state', 'action', 'reward', 'state+'))

    def Q(self, key):
        if not hasattr(self.mainQ, 'coefs_'):
            return 0
        key = (*key[0], self.actions.index(key[1]))
        return self.mainQ.predict([key])[0]


    def V(self, state):
        return max([self.Q(key=(state, a)) for a in self.actions])

    def targetV(self, state, env=None):
        if env.is_terminal(state):
            return 0
        return max([self.targetQ(key=(state, a)) for a in self.actions])

    def update(self, action, reward):
        self.cache = self.cache.append({'state':self.last_state, 'action':action, 'reward':reward, 'state+':self.state}, ignore_index=True)
        L = len(self.cache)
        if L > 800:
            self.cache.drop(np.arange(L-800+10))
        if L > 10 and self.n_steps % 3 == 2:
            self.learn()
        if self.n_steps % 15 == 14:
            self.updateQ()

    def get_samples(self, size=0.8):
        # state, action, reward, next_state ~ self.cache
        L = len(self.cache)
        size = int(size * L)
        inds = np.random.choice(L, size=size)
        states = self.cache.loc[inds, 'state'].values
        states = np.array([state for state in states])
        actions = self.cache.loc[inds, 'action']
        rewards = self.cache.loc[inds, 'reward'].values
        next_states = self.cache.loc[inds, 'state+'].values
        X = np.column_stack((states, actions))
        y = rewards + self.gamma * np.array([self.targetV(s, env) for s in next_states])
        return X, y


    def learn(self):
        X, y = self.get_samples()
        self.mainQ.fit(X, y)

    def updateQ(self):
        self.targetQ.coefs_ = self.mainQ.coefs_
        self.targetQ.intercepts_ = self.mainQ.intercepts_

