#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gym

gym.register(
    id='GridWorld-v1',
    entry_point='simple_grid1:MyGridWorld',
    max_episode_steps=200,
    reward_threshold=100.0
    )

from skinner import FiniteSet
from objects import Robot, NeuralRobot


class MyRobot(Robot):
    # action_space = Discrete(4)
    action_space = FiniteSet('news')

    length, width = 15, 45
    color = (0, 0.4, .7)

    @property
    def position(self):
        return self.state

    def _next_state(self, state, action):
        """transition function
        
        Arguments:
            state -- state before action
            action -- the action selected by the agent
        
        Returns:
            new state
        
        Raises:
            Exception -- invalid action
        """
        if action=='e':
            next_state = (state[0]+1, state[1])
        elif action=='w':
            next_state = (state[0]-1, state[1])
        elif action=='s':
            next_state = (state[0], state[1]-1)
        elif action=='n':
            next_state = (state[0], state[1]+1)
        else:
            raise Exception('invalid action!')
        if self.env.collide(next_state):
            next_state = state
        return next_state


    def _get_reward(self, state0, action, state1):
        """reward function
        
        called in step method
        
        Arguments:
            state0 -- state before action
            action -- the action
            state1 -- state after action
        
        Returns:
            number -- reward
        """
        
        if state1 in self.env.TRAPS:
            r = -100
        elif state1 in self.env.DEATHTRAPS:
            r = -200
        elif state1 == self.env.GOLD:
            r = 100
        elif state0 == state1:
            r = -2
        else:
            r = -1
        return r

    def reset(self):
        super(MyRobot, self).reset()
        self.state = (1, self.env.n_rows)


agent = MyRobot(alpha=0.7, gamma=0.9, epsilon=0.1)


if __name__ == '__main__':
    env = gym.make('GridWorld-v1', agent=agent)
    env.seed()
    env.demo(n_epochs=100)
