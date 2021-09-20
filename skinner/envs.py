#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gym
from gym.envs.classic_control import rendering

class BaseEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    history = None
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
            # draw the backgroud and objects of the env 
            self.create_viewer()       
            self.draw_background()
            self.draw_objects()

        return self.viewer.render(return_rgb_array=mode=='rgb_array')

    def post_process(self, *args, **kwargs):
        # postprocess after one train epoch
        pass

    def pre_process(self, *args, **kwargs):
        pass

    def begin_process(self, *args, **kwargs):
        self.pre_process(*args, **kwargs)

    def end_process(self):
        # postprocess after all train epochs
        if hasattr(self.history, 'to_csv'):
            self.history.to_csv(f'history.csv')

    def demo(self, n_epochs=200, n_steps=None, history=None):
        # business process of RL
        import time
        if n_steps is None:
            n_steps = self.max_steps
        self.begin_process()
        for i in range(n_epochs):
            self.reset()
            self.render()
            self.epoch = i
            for k in range(n_steps):
                if self.is_terminal():
                    break
                time.sleep(0)
                self.step()
                self.render()
            self.post_process()
        self.end_process()
        self.close()

    def draw_objects(self):
        for _, obj in self.objects.items():
            obj.draw(self.viewer)

    def reset(self):
        for _, obj in self.objects.items():
            obj.reset()

    def step(self):
        """A single step of the change of the whole environment;
        It is the core code of RL;
        It will call step methods of its objects iteratively
        """
        for _, obj in self.objects.items():
            obj.step()

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

    # def reset(self):
    #     self.agent.reset()

    def step(self):
        """A single step of the itertion of the env.
        Most important part is to call step method of the unique agent.
        """
        is_terminal = self.is_terminal()
        if not is_terminal:
            self.agent.step()

    def post_process(self):
        self.agent.post_process()

    def end_process(self):
        if hasattr(self.history, 'to_csv'):
            self.history.to_csv(f'history-{self.agent.name}.csv')

    