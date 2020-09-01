# skinner
Skinner, a new framework of reinforcement learning by Python

It is built for the beginner of RL.



It is under development, the APIs are not designed perfectly, but runs stably. For grid worlds, it is mature enough.



Enjoy `skinner`!

![](rat.gif)

## Requrements

- gym
- numpy

## Download

download from github, or pypi by pip command `pip install skinner`.

## Design

We consider the **observer design pattern**. The env and agents in it observe each other generally. The agents observe the env to how to act and got the reward, env observe the agents and other objects to render the viewer and record the information.

## Feature

so easy

## Use

### Quick start

run `demo.py` in examples. There are other examples: `demo1.py, demo2.py`.

Also, one could watch animations in [`bilibili`](https://www.bilibili.com/video/bv1ca4y1E7Dr)



### Examples

The author make 3 examples. users are suggested to review the codes. Define objects in `objects.py`, define new envs in `simple_grid.py` then write a demonstration programming in a script (see `demo.py`).

### Define envs

If you just want to build a simple env, then the following is an option, a grid world.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Demo of RL

An env with some traps and a gold.
"""

from skinner import *
from gym.envs.classic_control import rendering

from objects import *

class MyGridWorld(GridMaze, SingleAgentEnv):
    """Grid world
    
    A robot playing the grid world, tries to find the golden (yellow circle), meanwhile
    it has to avoid of the traps(black circles)
    Extends:
        GridMaze: grid world with walls
        SingleAgentEnv: there is only one agent
    """
    
    # configure the env
    
    # get the positions of the objects (done automatically)
    CHARGER = ...
    TRAPS = ...
    DEATHTRAPS = ...
    GOLD = ...

    def __init__(self, *args, **kwargs):
        super(MyGridWorld, self).__init__(*args, **kwargs)
        self.add_walls(conf['walls'])
        self.add_objects((*traps, *deathtraps, charger, gold))

    # Define the condition when the demo of rl will stop.
    def is_terminal(self):
        return self.agent.position in self.DEATHTRAPS or self.agent.position == self.GOLD or self.agent.power<=0

    def is_successful(self):
        return self.agent.position == self.GOLD

    # Following methods are not necessary, that only for recording the process of rl
    def post_process(self):
        if self.is_successful():
            self.history['n_steps'].append(self.agent.n_steps)
        else:
            self.history['n_steps'].append(self.max_steps)
        self.history['reward'].append(self.agent.total_reward)
        self.agent.post_process()

    def pre_process(self):
        self.history['n_steps'] = []
        self.history['reward'] = []

    def end_process(self):
        import pandas as pd
        data = pd.DataFrame(self.history)
        data.to_csv('history.csv')


```



#### Configure env and its objects

see `conf.yaml` for an example. The object classes would be defined in `objects.py`.

```yaml
# Grid Maze: 
# n_cols * n_rows: size of the maze, the number of squares
# edge: the length of the edge of each square
# walls: the positions of walls as the components of the environment


## number of grids
n_cols: 7
n_rows: 7
## size of every grid
edge: 80


## positions of walls
walls: !!set
  {
  !!python/tuple [2, 6],
  !!python/tuple [3, 6],
  ...
  !!python/tuple [4, 2]}


## objects in environment (excluding the agent)
## traps, not terminal
traps: !!python/object:objects.ObjectGroup
  name: 'traps'
  members:
    - !!python/object:objects.Trap
      position: !!python/tuple [3, 5]
      color: [1,0.5,0]
      size: 30
    - !!python/object:objects.Trap
      position: !!python/tuple [1, 3]
      color: [1,0.5,0]
      size: 30
    - !!python/object:objects.Trap
      position: !!python/tuple [7, 1]
      color: [1,0.5,0]
      size: 30

## deathtraps, terminal
deathtraps: !!python/object:objects.ObjectGroup
  name: 'deathtraps'
  members:
    - !!python/object:objects.DeathTrap
      position: !!python/tuple [6, 5]
      color: [.8,0,0.5]
      size: 35

    - !!python/object:objects.DeathTrap
      position: !!python/tuple [2, 1]
      color: [.8,0,0.5]
      size: 35

## gold, terminal
gold: !!python/object:objects.Gold
  name: 'gold'
  position: !!python/tuple
    [7, 7]
  color: [1,0.8,0]
  size: 30
```



### Define objects

1. the shape of object (circle by default)
2. the method to plot (don't override it, if the shape is simple)

```python
class _Object(Object):
    props = ('name', 'position', 'color', 'size')
    default_position=(0, 0)  # set default value to help you reducing the codes when creating an object

class Gold(_Object):
    def draw(self, viewer):
        '''this method is the most direct to determine how to plot the object
        You should define the shape and coordinate
        '''
        ...

class Charger(_Object):
    def create_shape(self):
        '''redefine the shape, here we define a squre with edges length of 40.
        The default shape is a circle
        '''
        a = 20
        self.shape = rendering.make_polygon([(-a,-a), (a,-a), (a,a), (-a,a)])
        self.shape.set_color(*self.color)
```



### Define agents

1. transition function $f(s,a)$
2. reward function $r(s,a,s')$

```python
from skinner import *

class MyRobot(StandardAgent):
    actions = Discrete(4)
    
    # define the shape
    size = 30
    color = (0.8, 0.6, 0.4)

    def _reset(self):
        # define the initial state
        ...
        
    def _next_state(self, state, action):
        """transition function: s, a -> s'
        """
        ...


    def _get_reward(self, state0, action, state1):
        """reward function: s,a,s'->r
        """
        ...


# define parameters
agent = MyRobot(alpha = 0.3, gamma = 0.9)
```


## Example

### codes

see scripts in `examples`

### results

![](performance.png)



## Commemoration

In memory of [B. F. Skinner](https://www.bfskinner.org/) (1904-1990), a great American psychologist. The RL is mainly inspired by his behaviorism. There are many contributors in the history of behaviorist psychology, he may be the most famous one.

 ![](skinner.jpg)
