#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections.abc import Iterable
from gym.spaces import Discrete

class FiniteSet(Discrete):
    r"""A discrete space in :math:`\{ a,b,c ... \}`. 
    Example::
        >>> FiniteSet('news')
    """
    def __init__(self, actions):
        assert isinstance(actions, Iterable)
        self.actions = tuple(actions)
        super(FiniteSet, self).__init__(n, np.int64)

    def sample(self):
        return self.np_random.choice(self.actions)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.char in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return x in self.actions
        return 0 <= as_int < self.n

    def __repr__(self):
        return "FiniteSet(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, FiniteSet) and self.n == other.n