# skinner
Skinner, a new framework of reinforcement learning by Python



## Requrements

- gym
- numpy

## Use

### Define agents

```python
from skinner import *

class MyRobot(StandardAgent):
    # actions = Discrete(4)
    # define parameters
    alpha = 0.3
    gamma = 0.9
    # define the shape
    size = 30
    color = (0.8, 0.6, 0.4)

    def _next_state(self, state, action):
        """transition function: s, a -> s'
        """
        ...


    def _get_reward(self, state0, action, state1):
        """reward function: s,a,s'->r
        """
        ...

    def reset(self):
        super(MyRobot, self).reset()
        ...


agent = MyRobot()
```



## Example

### codes

### results

![](performance.png)



## Commemoration

In memory of B. F. Skinner(1904-1990), a great American psychologist

 ![](skinner.jpg)