#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gym

gym.register(
    id='GridWorld-v1',
    entry_point='simple_grid:MyGridWorld',
    max_episode_steps=500,
    reward_threshold=1000
    )

from objects import Robot, NeuralRobot, BoltzmannRobot, BayesRobot


class MyRobot(BayesRobot):

    def detect(self):
        right = left = top = bot = 0
        if self.position[0] >= self.env.n_cols:
            right = 1
        elif self.position[0] <= 1:
            left = 1
        if self.position[1] >= self.env.n_rows:
            top = 1
        elif self.position[1] <= 1:
            bot = 1
        if (self.position[0] + 1, self.position[1]) in self.env.TRAPS | self.env.DEATHTRAPS | self.env.WALLS:
            right = 2
        if (self.position[0] - 1, self.position[1]) in self.env.TRAPS | self.env.DEATHTRAPS | self.env.WALLS:
            left = 2
        if (self.position[0] + 1, self.position[1]) in self.env.TRAPS | self.env.DEATHTRAPS | self.env.WALLS:
            top = 2
        if (self.position[0] - 1, self.position[1]) in self.env.TRAPS | self.env.DEATHTRAPS | self.env.WALLS:
            bot = 2
        if (self.position[0] + 1, self.position[1]) == self.env.GOLD:
            right = 3
        if (self.position[0] - 1, self.position[1]) == self.env.GOLD:
            left = 3
        if (self.position[0] + 1, self.position[1]) == self.env.GOLD:
            top = 3
        if (self.position[0] - 1, self.position[1]) == self.env.GOLD:
            bot = 3
        self.state = (self.state[0], self.state[1], left, right, top, bot)


    
    def _reset(self):
        self.state = 1, self.env.n_rows, 0,0,0,0
        self.detect()

    def next_state(self, action):
        """
        self.__next_state: state transition method
        function: state, action -> state
        """
        self.n_steps +=1
        self.last_state = self.state
        self.state = self._next_state(self.last_state, action)
        self.detect()
        

    def _next_state(self, state, action):
        """Transition function
        
        Arguments:
            state -- state before action
            action -- the action selected by the agent
        
        Returns:
            new state
        
        Raises:
            Exception -- invalid action
        """

        if action == 'e':
            next_state = state[0]+1, state[1], *state[2:]
        elif action == 'w':
            next_state = state[0]-1, state[1], *state[2:]
        elif action == 'n':
            next_state = state[0], state[1]+1, *state[2:]
        elif action == 's':
            next_state = state[0], state[1]-1, *state[2:]
        else:
            raise Exception('invalid action!')
        if self.env.collide(next_state[:2]):
            next_state = state
        return next_state


    def _get_reward(self, state0, action, state1):
        """Reward function
        
        called in step method
        
        Arguments:
            state0 -- state before action
            action -- the action
            state1 -- state after action
        
        Returns:
            number -- reward
        """
        
        if state1[:2] in self.env.TRAPS:
            r = -20
        elif state1[:2] in self.env.DEATHTRAPS:
            r = -50
        elif state1[:2] == self.env.GOLD:
            r = 100
        elif state0[:2] == state1:
            r = -2
        else:
            r = -1
        return r


agent = MyRobot(alpha=0.75, gamma=0.95, epsilon=0.1)


if __name__ == '__main__':

    env = gym.make('GridWorld-v1')
    env.config('config.yaml')
    env.add_agent(agent)
    env.seed()
    env.demo(n_epochs=300, n_steps=500)
