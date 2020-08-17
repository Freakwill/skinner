#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""configuration of the env

the number/size of grids
traps
gold/goal
"""

## 格子数目
M, N = 7, 7
## 窗口大小
edge = 100


## 陷阱
TRAPS = {
(1,2),
(5,3),
(2,4)
}

## 死亡陷阱
DEATHTRAPS = {
(4,4),
(3,3),
(7,1)
}

## 金币
GOLD = (3,1)