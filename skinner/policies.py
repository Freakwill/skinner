#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import random
import numpy as np
from scipy.stats import rv_discrete
from scipy.special import softmax


def greedy(state, action_space, Q, epsilon=0.01):
    # greedy policy
    if random() < epsilon:
        return action_space.sample()

    k = np.argmax([Q((state, a)) for a in action_space])
    return action_space[k]


def boltzmann(state, action_space, Q, temperature=1, epsilon=0.01):
    # greedy policy
    qs = np.array([Q((state, a)) for a in action_space])
    if random() < epsilon:
        p = softmax(qs / temperature)
        d = rv_discrete(values=(np.arange(len(qs)), p))
        return action_space[d.rvs()]

    k = np.argmax(qs)
    return action_space[k]


def post_proba(Q, s, action_space, temperature=1):
    """Posteria proba of c in action_space
    {p(a|s)} ~ softmax(Q(s,a))
    
    Arguments:
        Q {dict} -- Q table
        s {array} -- state
        action_space {Space|array} -- space of action_space or an array of action_space
    
    Keyword Arguments:
        temperature {number} -- scaling Q value (default: {1})
    
    Returns:
        array with |action_space| elements
    """

    return softmax([temperature*Q.get((s, a), 0) for a in action_space])


def joint_proba_i(Q, si, i, action_space, states=None, temperature=1, ps=None):
    # joint proba of a in action_space under Xi=xi
    if states is None:
        states = set(s for s, a in Q.keys())
    return np.mean([post_proba(Q, s, action_space, temperature) * ps.get(s, 0) if s[i]==si else np.ones(action_space.n)*0.0001 for s in states], axis=0)


def ez_proba_i(Q, si, i, action_space, states=None, temperature=1):
    # easy version of joint_proba_i
    return np.mean([post_proba(Q, s, action_space, temperature) for s in states if s[i]==si], axis=0)


def predict_proba(Q, s, action_space, states=None, temperature=1, pa=None, ps=None):
    # pa: priori proba of actions
    P = np.array([joint_proba_i(Q, si, i, action_space, states=states, temperature=temperature, ps=ps) for i, si in enumerate(s)])
    pa_s = temperature * np.sum(np.log(P), axis=0) + (1-len(s)) * np.log([pa[a] for a in action_space])
    return pa_s / np.sum(pa_s)

def predict(Q, s, states=None, temperature=1, pa=None):
    pa_s = predict_proba(Q, s, states, temperature, pa)
    return np.argmax(pa_s)

def generate(Q, s, action_space, states=None, temperature=1, pa=None, ps=None):
    pa_s = predict_proba(Q, s, action_space, states, temperature, pa=pa, ps=ps)
    rv = rv_discrete(values=(np.arange(len(pa_s)), pa_s))
    return action_space[rv.rvs()]


def bayes(state, action_space, Q, temperature=1, epsilon=0.01, pa=None, ps=None):
    """
    if (s, a) is not in Q esp. s is not in Q (any (s, a) is not in Q),
    then predict p(a|s) with Naive Bayes method.
    else select an action with Boltzmann method or epsilon-greedy method.
    
    Arguments:
        state {tuple} -- the state of the agent
        action_space {Space} -- action space of the agent
        Q {dict} -- Q table
    
    Keyword Arguments:
        temperature {number} -- temperature factor (default: {1})
        epsilon {number} -- epsilon for greedy algorithm (default: {0.01})
    """ 

    if not [s for s, a in Q.keys() if s==state] and pa and ps:
        return generate(Q, state, action_space, states=None, temperature=temperature, pa=pa, ps=ps)
    else:
        qs = np.array([Q.get((state, a),0) for a in action_space])
        if random() < epsilon:
            p = softmax(qs / temperature)
            q = predict_proba(Q, state, action_space, states=None, temperature=1, pa=pa, ps=ps)
            r = np.sum([p[i] for i, a in enumerate(action_space) if (state, a) not in Q]) / np.sum([q[i] for i, a in enumerate(action_space) if (state, a) not in Q])
            for i, a in enumerate(action_space):
                if (state, a) not in Q:
                    p[i]=q[i] * r
            d = rv_discrete(values=(np.arange(len(qs)), p))
            return action_space[d.rvs()]

        k = np.argmax(qs)
        return action_space[k]
