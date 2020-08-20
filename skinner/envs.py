#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gym
from gym.envs.classic_control import rendering

class MyEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    history = {}
    max_steps = 150
    
    def is_terminal(self):
        raise NotImplementedError

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:  
            self.create_viewer()
            # draw the backgroud of the env           
            self.draw_background()
            # draw the objects in env
            self.draw_objects()

        if self.state is None:
            return None
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def post_process(self, *args, **kwargs):
        pass

    def pre_process(self, *args, **kwargs):
        pass

    def end_process(self, *args, **kwargs):
        pass

    def demo(self, n_epochs=100, history=None):
        import time
        # demo of RL
        self.pre_process()
        for i in range(n_epochs):
            self.reset()
            self.render()
            self.epoch = i
            for k in range(self.max_steps):
                time.sleep(0.001)
                self.step()
                self.render()
                done = self.is_terminal()
                if done:
                    break
            self.post_process()
        self.end_process()
        self.close()


class MultiAgentEnv(MyEnv):
    def __init__(self, objects={}):
        self.objects = objects
        self.viewer = None
        self.state = None

class SingleAgentEnv(MyEnv):
    def __init__(self, agent):
        self.agent = agent
        self.viewer = None

    def reset(self):
        self.agent.reset()

    @property
    def state(self):
        return self.agent.state

    @property
    def last_state(self):
        return self.agent.last_state

    def get_reward(self, action):
        return self._get_reward(self.last_state, action, self.state)

    def step(self):
        is_terminal = self.is_terminal()
        if is_terminal:
            return
        r = self.agent.step(self)
        return self.agent, r, is_terminal, {}

    