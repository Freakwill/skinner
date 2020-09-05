#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gym

gym.register(
    id='GridWorld-v1',
    entry_point='simple_grid:MyGridWorld3',
    max_episode_steps=200,
    reward_threshold=1000
    )

from objects import Robot, NeuralRobot


class MyRobot(Robot):
    init_power = 25

    @property
    def flag1(self):
        return self.state[2]

    @property
    def flag2(self):
        return self.state[3]

    @property
    def flag3(self):
        return self.state[4]

    @property
    def power(self):
        return self.state[5]
    

    def _reset(self):
        self.state = 1, self.env.n_rows, 0, 0, 0, self.init_power

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
            next_state = (state[0]+1, state[1], *state[2:])
        elif action == 'w':
            next_state = (state[0]-1, state[1], *state[2:])
        elif action == 's':
            next_state = (state[0], state[1]-1, *state[2:])
        elif action == 'n':
            next_state = (state[0], state[1]+1, *state[2:])
        else:
            raise Exception('invalid action!')
        if self.env.collide(next_state[:2]):
            next_state = state
        if next_state[:2] == self.env.BUTTON1:
            next_state = (next_state[0], next_state[1], 1, next_state[3], next_state[4], next_state[5])
        elif next_state[:2] == self.env.BUTTON2:
            next_state = (next_state[0], next_state[1], next_state[2], 1, next_state[4], next_state[5])
        elif next_state[:2] == self.env.BUTTON3:
            next_state = (*next_state[:4], 1, next_state[5])
        elif next_state[:2] == self.env.CHARGER:
            next_state = (*next_state[:5], 50)
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
        elif state1[:2] == self.env.DEATHTRAP1:
            if state1[2]:
                r = -1
            else:
                r = -30
        elif state1[:2] == self.env.DEATHTRAP2:
            if state1[3]:
                r = -1
            else:
                r = -30
        elif state1[:2] == self.env.DEATHTRAP3:
            if state1[3]:
                r = -1
            else:
                r = -30
        elif state1[:2] == self.env.DEATHTRAP4:
            r = -30
        elif state1[:2] == self.env.GOLD:
            r = 20
        elif state0[:2] == state1[:2]:
            r = -2
        else:
            r = -1
        return r


if __name__ == '__main__':

    env = gym.make('GridWorld-v1')
    env.config('config3.yaml')
    agent = MyRobot(alpha=0.7, gamma=0.9, epsilon=0.1)
    env.add_agent(agent)
    env.seed()
    env.demo(n_epochs=200)
