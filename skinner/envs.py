#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gym
from gym.envs.classic_control import rendering

class BaseEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    history = {}
    max_steps = 200
    viewer = None

    def __init__(self, objects={}):
        self._objects = objects

    @property
    def objects(self):
        return self._objects

    def add_objects(self, objs):
        self._objects.update(objs)
        for _, obj in objs.items():
            obj.env = self
    
    
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

    def demo(self, n_epochs=200, history=None):
        import time
        # demo of RL
        self.pre_process()
        for i in range(n_epochs):
            self.reset()
            self.render()
            self.epoch = i
            for k in range(self.max_steps):
                time.sleep(.001)
                self.step()
                self.render()
                done = self.is_terminal()
                if done:
                    break
            self.post_process()
        self.end_process()
        self.close()

    def draw_objects(self):
        for _, obj in self.objects.items():
            obj.draw(self.viewer)

    def reset(self):
        for _, obj in self.objects.items():
            obj.reset()

class MultiAgentEnv(BaseEnv):
    pass


class SingleAgentEnv(BaseEnv):
    def __init__(self, agent=None):
        if agent:
            self.add_objects({'agent': agent})

    @property
    def agent(self):
        return self.objects['agent']

    def add_agent(self, agent):
        self.add_objects({'agent': agent}) 

    def reset(self):
        self.agent.reset()

    @property
    def state(self):
        return self.agent.state

    @property
    def last_state(self):
        return self.agent.last_state

    def step(self):
        """A single step of the itertion of the env.
        Most important part is to call step method of the unique agent.
        """
        is_terminal = self.is_terminal()
        if is_terminal:
            return self.agent, r, True, {}
        else:
            r = self.agent.step()
            return self.agent, r, False, {}

    