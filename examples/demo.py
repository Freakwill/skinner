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
from objects import Robot


class MyRobot(Robot):
    # actions = Discrete(4)
    actions = FiniteSet('news')

    alpha = 0.3
    gamma = 0.9

    size = 30
    color = (0, 0.1, 1)

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
        last_state = state
        if action=='e':
            if state[0]<=self.env.n_rows-1:
                state = (state[0]+1, state[1])
        elif action=='w':
            if state[0]>=2:
                state = (state[0]-1, state[1])
        elif action=='s':
            if state[1]>=2:
                state = (state[0], state[1]-1)
        elif action=='n':
            if state[1]<=self.env.n_cols-1:
                state = (state[0], state[1]+1)
        else:
            raise Exception('invalid action!')
        if self.collide(state):
            state = last_state
        return state

    def collide(self, state):
        for wall in self.env.walls:
            if wall == state:
                return True
        else:
            return False


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
            return -1
        elif state1 in self.env.DEATHTRAPS:
            return -2
        elif state1 == self.env.GOLD:
            return 3
        elif state0 == state1:
            return -0.2
        else:
            return -0.05

    def reset(self):
        super(MyRobot, self).reset()
        self.state = 1, self.env.n_rows


agent = MyRobot()


if __name__ == '__main__':
    env = gym.make('GridWorld-v1', agent=agent)
    env.seed()
    env.demo()
