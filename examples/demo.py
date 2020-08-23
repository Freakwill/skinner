#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gym

gym.register(
    id='GridWorld-v1',
    entry_point='simple_grid:MyGridWorld',
    max_episode_steps=200,
    reward_threshold=100.0
    )

from skinner import FiniteSet
from objects import Robot, NeuralRobot


class MyRobot(Robot):
    # action_space = Discrete(4)
    action_space = FiniteSet('news')

    alpha = 0.5
    gamma = 0.8
    epsilon = 0.1

    size = 30
    color = (0, 0.1, 1)

    init_power = 20

    @property
    def power(self):
        return self.state[2]

    @property
    def position(self):
        return self.state[:2]

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
            next_state = (state[0]+1, state[1], state[2])
        elif action=='w':
            next_state = (state[0]-1, state[1], state[2])
        elif action=='s':
            next_state = (state[0], state[1]-1, state[2])
        elif action=='n':
            next_state = (state[0], state[1]+1, state[2])
        else:
            raise Exception('invalid action!')
        if self.env.collide(next_state[:2]):
            next_state = state
        else:
            if next_state[:2] == self.env.CHARGER:
                next_state = *next_state[:2], 50
            else:
                next_state = *next_state[:2], next_state[2]-1
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
        
        r = 0
        if state0[-1] <=4:
            if state1[:2] == self.env.CHARGER:
                r += .2
            else:
                r -= .2
        else:
            if state1[:2] == self.env.CHARGER:
                r -= .1
        if state1[:2] in self.env.TRAPS:
            r -= 10
        elif state1[:2] in self.env.DEATHTRAPS:
            r -= 20
        elif state1[:2] == self.env.GOLD:
            r += 10
        elif state0[:2] == state1[:2]:
            r -= 0.2
        else:
            r -= 0.1
        return r

    def reset(self):
        super(MyRobot, self).reset()
        self.state = (1, self.env.n_rows, self.init_power)


agent = MyRobot()


if __name__ == '__main__':
    env = gym.make('GridWorld-v1', agent=agent)
    env.seed()
    env.demo(n_epochs=2000)
