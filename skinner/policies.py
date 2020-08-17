#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import choice, random
import numpy as np

def greedy(state, actions, Q, epsilon=0.02, default_action=None):
    # greedy policy
    action = default_action
    if random()<epsilon:
        return action

    k = np.argmax([Q((state, a)) for a in actions])
    return actions[k]