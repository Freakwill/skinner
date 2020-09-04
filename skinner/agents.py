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

    def is_terminal(self, key):
        raise NotImplementedError

    def update(self, key):
        raise NotImplementedError

    def step(self, env):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def reset(self):
        self._reset()
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

    def save(self, fname=None):
        import pickle, pathlib
        if self.name:
            pklPath = pathlib.Path(f'{self.name}.pkl')
        else:
            pklPath = pathlib.Path('agent.pkl')
        with open(pklPath, 'wb') as fo:
            pickle.dump(self, fo)

    def load(self, fname=None):
        import pickle, pathlib
        if fname:
            if isinstance(fname, str):
                fname = pathlib.Path(fname)
        elif self.name:
            pklPath = pathlib.Path(f'{self.name}.pkl')
        else:
            pklPath = pathlib.Path('agent.pkl')
        if pklPath.exists():
            with open(pklPath, 'rb') as fo:
                return pickle.load(fo)
        else:
            raise IOError('Do not find the pickle file!')


class StandardAgent(BaseAgent):

    epoch = 1

    def __init__(self, QTable={}, init_state=None, gamma=0.9, alpha=0.5, epsilon=0.1):
        """[summary]
        
        [description]
        
        Keyword Arguments:
            QTable {dict} -- Q Table (default: {{}})
            init_state {[type]} -- initial state (default: {None})
            gamma {number} -- discount rate (default: {0.9})
            alpha {number} -- learning rate (default: {0.5})
            epsilon {number} -- factor for selecting actions (default: {0.1})
        """
        self.QTable = QTable
        self.VTable = {}
        for key, value in QTable.items():
            state, action = key
            if state not in self.VTable:
                self.VTable[state] = self.V(state)
            elif self.VTable[state] < q:
                self.VTable[state] = q
        self.last_state = None
        self.init_state = init_state
        self.state = init_state
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def select_action(self):
        return greedy(self.state, self.action_space, self.Q, self.epsilon)

    def step(self):
        """
        A single step of iteration

        1. select an action
        2. execute the action and transitate to the next state
        3. get reward from env after the action
        4. update the QTable or some structure presenting QTable
        """

        action = self.select_action()
        self.next_state(action)
        reward = self.get_reward(action)
        self.update(action, reward)

        # uncomment the following code to debug
        print(f'''action: {action}
        reward: {reward}
        state: {self.last_state} => {self.state}
        Q corrected: {[(action, self.Q((self.last_state, action))) for action in self.action_space]}
        ''')

        return reward


    def Q(self, key):
        """Predict Q value of key based on current QTable

        key: tuple of (state, action)

        Return 0 by default, if key is not in QTable
        """
        return self.QTable.get(key, 0)

    def V(self, state):
        # if state in self.VTable:
        #     return self.VTable[state]
        return self._V(state)

    def _V(self, state):
        return max([self.Q(key=(state, a)) for a in self.action_space])

    def update(self, action, reward):
        """Update the QTable by Q Learning

        The core code of the algorithm
        
        Arguments:
            action {[type]} -- the selected action
            reward {number} -- the reward given by the env
        """

        key = self.last_state, action
        if key not in self.QTable:
            self.QTable[key] = self.Q(key)
        v = self.V(self.state)
        self.QTable[key] += self.alpha * (reward + self.gamma * v - self.QTable[key])

        # q = self.QTable[key]
        # if self.last_state in self.VTable:
        #     if self.VTable[self.last_state] < q:
        #         self.VTable[self.last_state] = q
        # elif q>=0:
        #     self.VTable[self.last_state] = q


    def draw(self, viewer, flag=True):
        if flag:
            super(StandardAgent, self).draw(viewer)
        else:
            self.transform.set_translation(*self.coordinate)

    def post_process(self, *args, **kwargs):
        # post process after a single step
        self.epsilon *= .9999
        self.alpha *= .9999

    def save_QTable(self, fname=None):
        with open('Q Table.txt', 'wb') as fo:
            fo.write(str(self.QTable))


class SarsaAgent(StandardAgent):

    def reset(self):
        self._reset()
        self.n_steps = 0
        self.total_reward = 0
        self.action = self.select_action()

    def step(self):
        """
        A single step of iteration

        1. execute the action and transitate to the next state
        2. get reward from env after the action
        3. update the QTable or some structure presenting QTable with sarsa
        """
        
        self.next_state(self.action)
        reward = self.get_reward(self.action)
        self.update(self.action, reward)

        return reward

    def update(self, action, reward):
        """Update the QTable by Sarsa

        The core code of the algorithm
        
        Arguments:
            action {[type]} -- the selected action
            reward {number} -- the reward given by the env
        """
        key = self.last_state, action
        if key not in self.QTable:
            self.QTable[key] = self.Q(key)
        self.action = self.select_action()
        self.QTable[key] += self.alpha * (reward + self.gamma * self.Q((self.state, self.action)) - self.QTable[key])


class BoltzmannAgent(StandardAgent):
    similarity = None    # similarity of pairs of (state, action)
    temperature = 10

    def select_action(self):
        return boltzmann(self.state, self.action_space, self.Q, temperature=self.temperature, epsilon=self.epsilon)

    def post_process(self, *args, **kwargs):
        # post process after a single step
        super(BoltzmannAgent, self).post_process(*args, **kwargs)
        self.temperature *= 0.99


import pandas as pd
from sklearn.neural_network import *


class MLAgent(StandardAgent):

    flag = True

    def __init__(self, state=None, last_state=None, init_state=None, mainQ=None, targetQ=None, *args, **kwargs):
        super(MLAgent, self).__init__(*args, **kwargs)
        self.state = state
        self.last_state = last_state
        self.init_state = init_state
        self.mainQ = mainQ
        self.targetQ = targetQ
        self.epoch = 1
        self.gamma = 0.9
        self.alpha = 0.1
        self.epsilon = 0.2
        self.cache = pd.DataFrame(columns=('state', 'action', 'reward', 'state+'))
        self.max_cache = 500

    @classmethod
    def key2vector(cls, key):
        return *key[0], key[1]

    def Q(self, key):
        # Q value of main net
        if not hasattr(self.mainQ, 'coefs_'):
            return 0
        x = self.__class__.key2vector(key)
        return self.mainQ.predict([x])[0]

    def Q_(self, key):
        # Q value of target net
        try:
            x = self.key2vector(key)
            return self.targetQ.predict([x])[0]
        except:
            return 0


    def V(self, state):
        if self.env.is_terminal():
            return 0
        return max([self.Q(key=(state, a)) for a in self.action_space])

    def V_(self, state):
        if self.env.is_terminal():
            return 0
        return max([self.Q_(key=(state, a)) for a in self.action_space])

    def update(self, action, reward):
        self.cache = self.cache.append({'state':self.last_state, 'action':action, 'reward':reward, 'state+':self.state}, ignore_index=True)
        L = len(self.cache)
        if L > self.max_cache:
            self.cache.drop(np.arange(L-self.max_cache+10))
        if L > 100:
            if self.n_steps % 5 == 4:
                self.learn()
            if self.n_steps % 30 == 29:
                self.updateQ()

    def get_samples(self, size=0.1):
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
        y = rewards + self.gamma * np.array([self.V_(s) for s in next_states])
        return X, y

    def learn(self):
        X, y = self.get_samples()
        self.mainQ.fit(X, y)
        if self.flag:
            self.targetQ.fit(X,y)
            self.flag = False


class NeuralAgent(MLAgent):
    def __init__(self, *args, **kwargs):
        super(NeuralAgent, self).__init__(*args, **kwargs)
        self.mainQ = MLPRegressor(hidden_layer_sizes=(6,), max_iter=100, warm_start=True, learning_rate='adaptive')
        self.targetQ = MLPRegressor(max_iter=1)

    def updateQ(self):
        self.targetQ.coefs_ = self.mainQ.coefs_
        self.targetQ.intercepts_ = self.mainQ.intercepts_

