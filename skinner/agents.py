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

    def next_state(self, action):
        """
        self.__next_state: state transition method
        function: state, action -> state
        """
        self.n_steps +=1
        self.last_state = self.state
        self.state = self._next_state(self.last_state, action)

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

    def reset(self):
        self.n_steps = 0


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
        self.epsilon = 0.02

    def select_action(self, default_action=None):
        if default_action is None:
            default_action=choice(self.actions)
        return greedy(self.state, self.actions, self.Q, self.epsilon, default_action)

    def get_reward(self, env, action):
        return self._get_reward(env, self.last_state, action, self.state)

    def step(self, env):
        action = self.select_action()
        self.next_state(action)
        reward = self.get_reward(env, action)
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

    def draw(self, viewer):
        if viewer is None:
            super(StandardAgent, self).draw(viewer)
        else:
            self.transform.set_translation(**self.coordinate)

    def post_process(self, *args, **kwargs):
        self.epsilon = self.epsilon ** .99
        self.alpha = self.alpha ** .99

from sklearn.neural_network import *

class NeuralAgent(StandardAgent):

    def __init__(self, QTable, state=None, last_state=None, init_state=None):
        self.QTable = {}
        self.state = state
        self.last_state = last_state
        self.init_state = init_state
        self.mainQ = MLPRegressor(hidden_layer_sizes=(10,), max_iter=200)
        self.targetQ = MLPRegressor(hidden_layer_sizes=(10,))
        self.epoch = 1
        self.gamma = 0.9
        self.alpha = 0.1
        self.epsilon = 0.2
        self.cache = []

    def Q(self, key):
        try:
            key = (*key[0], self.actions.index(key[1]))
            return self.mainQ.predict([key])[0]
        except:
            return 0

    def V(self, state):
        return max([self.Q(key=(state, a)) for a in self.actions])

    def update(self, action, reward):
        # self.QTable[key] += self.alpha * (reward + self.gamma * self.V() - self.QTable.get(key, 0))
        action = self.actions.index(action)
        self.cache.append((self.last_state, action, reward, self.state))
        if len(self.cache)>200:
            self.cache.pop(np.random.randint(200))

    def get_samples(self, size=30):
        # state, action, reward, next_state ~ self.cache
        L = len(self.cache)
        inds = np.random.choice(L, size=size)
        states = np.array([state for state, _, _, _ in self.cache])[inds]
        actions = np.array([action for _, action, _, _ in self.cache])[inds]
        rewards = np.array([reward for _, _, reward, _ in self.cache])[inds]
        next_states = np.array([next_state for _, _, _, next_state in self.cache])[inds]
        X = np.column_stack((states, actions))
        y = rewards + self.gamma*np.array([self.V(s) for s in next_states])
        return X, y


    def learn(self):
        X, y = self.get_samples()
        self.mainQ.fit(X, y)

    def updateQ(self):
        self.targetQ.set_params(**self.mainQ.get_params())

