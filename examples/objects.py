#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from skinner import *
from config import *


def _coordinate(position):
    return position[0]*edge+edge//2, position[1]*edge+edge//2


class _Object(Object):

    def __init__(self, name='', state=None, position=(1,1), color=(0,0,0), size=30):
        self.position = position
        super(_Object, self).__init__(state, name, coordinate=_coordinate(position), color=color, size=size)

class Trap(_Object):
    '''[Summary for Class Trap]'''
    pass
        
class DeathTrap(Trap):
    '''[Summary for Class Trap]'''
    def __init__(self, name='', state=None, position=(1,1), color=(1,0,0), size=30):
        super(DeathTrap, self).__init__(state, name, position=position, color=color, size=size)
        

class Gold(_Object):
    def draw(self, viewer):
        super(Gold, self).draw(viewer)
        gold_hole = _Object(name='gold_hole', state=True, position=self.position, color=(1, 1, 1), size=15)
        gold_hole.draw(viewer)
       