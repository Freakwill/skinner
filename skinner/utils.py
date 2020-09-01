#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.special
import numpy as np

def post_proba(Q, x, actions, T=1):
    """Posteria proba of c in actions
    {p(a|x)} ~ softmax(Q(x,a))
    
    Arguments:
        Q {dict} -- Q table
        x {array} -- state
        actions {array|list} -- array of actions
    
    Keyword Arguments:
        T {number} -- scaling Q value (default: {1})
    
    Returns:
        array with |actions| elements
    """
    return scipy.special.softmax(T*Q[x, a] for a in actions)


def joint_proba(Q, x, actions, T=1):
    # joint proba of x, c for c in actions
    return post_proba(Q, xi, actions, T) * proba(x)

def joint_proba_i(Q, xi, i, states, actions, T=1):
    # joint proba of c in actions under Xi=xi
    return np.mean([post_proba(Q, xi, actions, T) * proba(x) for x in states if x[i]==xi], axis=0)


def ez_proba_i(Q, xi, i, states, actions, T=1):
    # easy version of joint_proba_i
    return np.mean([post_proba(Q, xi, actions, T) for x in states if x[i]==xi], axis=0)


def post_proba(Q, x, states, T=1):
    return scipy.special.softmax(join_proba_i(Q, xi, i, states, T))

def predict_proba(Q, x, states, T=1):
    P = np.array([joint_proba_i(Q, xi, i, states, T) for xi in x])
    A = T * np.sum(P, axis=0) + (1-n) * np.log(pp)
    return A

def predict(Q, x, states, T=1):
    A = predict_proba(Q, x, states, T)
    return np.argmax(A)

