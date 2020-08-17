#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""强化学习演示程序V2.0

在这个格子世界的一个机器人要去寻找金子（黄色圆圈），同时需要避免很多陷阱（黑色圆圈）

主要方法:
step: 每次主体与环境互动的流程
    调用get_reward与nex_state
get_reward: 每次互动获得回报 r=g(s,a,s')
next_state: 转态迁移 s'=f(s,a)
"""

from skinner import *
from gym.envs.classic_control import rendering

from config import *
from objects import *

screen_width = edge*(M+2)
screen_height = edge*(N+2)

class GridWorld(SingleAgentEnv):
    """Grid world 格子世界
    
    A robot playing the grid world, tries to find the golden (yellow circle), meanwhile
    it has to avoid of the traps(black circles)
    在这个格子世界的一个机器人要去寻找金子（黄色圆圈），同时需要避免很多陷阱（黑色圆圈）
    
    Extends:
        gym.Env
    
    Variables:
        metadata {dict} -- configuration of rendering
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, agent):
        super(GridWorld, self).__init__(agent)
        self.terminate_states = DEATHTRAPS | {GOLD}


    def _get_reward(self, state, action, next_state):
        """回报函数
        
        被step方法调用
        
        Arguments:
            state -- 动作之前的状态
            action -- 动作
            next_state -- 动作之后的状态
        
        Returns:
            number -- 回报值
        """
        if next_state in TRAPS:
            return -1
        elif next_state in DEATHTRAPS:
            return -2
        elif next_state in {GOLD}:
            return 3
        elif state == next_state:
            return -0.2
        else:
            return -0.1

    def is_terminal(self):
        return self.state in self.terminate_states

    def is_successful(self):
        return self.state == GOLD

    def reset(self):
        self.agent.reset()
        self.agent.state = 1, N

    def post_process(self):
        if self.is_successful():
            self.history['n_steps'].append(self.agent.n_steps)
        else:
            self.history['n_steps'].append(self.max_steps)
        self.agent.post_process()

    def pre_process(self):
        self.history['n_steps'] = []

    def end_process(self):
        import pandas as pd
        data = pd.DataFrame(self.history)
        data.to_csv('history.csv')

    def draw_background(self):
        # grid world           
        for k in range(M+1):
            line = rendering.Line(((1+k)*edge, edge), ((1+k)*edge, (N+1)*edge))
            line.set_color(0,0,0)
            self.viewer.add_geom(line)
        for k in range(N+1):
            line = rendering.Line((edge, (1+k)*edge), ((M+1)*edge, (1+k)*edge))
            line.set_color(0,0,0)
            self.viewer.add_geom(line)


    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.draw_background()
            # traps
            for trap_pos in TRAPS:
                trap = Trap(name='trap', state=True, position=trap_pos, color=(0,0,0), size=30)
                trap.draw(self.viewer)
            # deathtraps
            for trap_pos in DEATHTRAPS:
                trap = DeathTrap(name='death trap', state=True, position=trap_pos, color=(1,0,0), size=30)
                trap.draw(self.viewer)
            # gold
            gold = Gold(name='gold', state=True, position=GOLD, color=(1, 0.9, 0), size=30)
            gold.draw(self.viewer)

            # robot
            self.robot= rendering.make_circle(30)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0.8, 0.6, 0.4)
            self.viewer.add_geom(self.robot)

        if self.state is None:
            return None
        self.robotrans.set_translation(edge*self.state[0]+50, edge*self.state[1]+50)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
