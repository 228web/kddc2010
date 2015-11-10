# -*- coding: utf-8 -*-
"""
Created on Mon Nov 09 19:20:50 2015

@author: John
"""

import numpy as np
import numpy.random as rand

def generate_hmm_dat(n,p):
    """
    Loaded vs. unloaded die example
    
    Inputs
    ------
    n : int
        number of points to make
    p : float
        probability of transitioning to opposite state
        
    Returns
    -------
    x : ndarray
        nx1 array of dice roll value give fair or loaded
    y : ndarray
        (n+1)x1 boolean array of [1,0] = [fair, loaded] state
    """
    x = np.ones(n)
    y = np.zeros(n+1)
    y[0] = rand.choice(2)
    
    if p > 1 or p < 0:
        p = 0
        print "Invalid probability, p set to 0, all fair."
    
    for k in range(n):
        yNow = rand.rand()
        if yNow > p:
            y[k+1] = not y[k-1]
        else:
            y[k] = y[k-1]
            
        if y[k] == 1:
            x[k] = rand.choice(6)
        else:
            x[k] = rand.choice(6,p=[.1,.1,.1,.1,.1,.5])
            
    return x,y
        
def general_hmm_dat(n,start_p, trans_p, emit_p):
    """
    Loaded vs. unloaded die example
    
    Inputs
    ------
    n : int
        number of points to make
    start_p : ndarray
        kx1 probabilities for starting in any hidden model state
    trans_p : ndarray
        kxk transition probabilities of hidden model
    emit_p : float
        kxd probability of emitting value from given hidden model state
        
    Returns
    -------
    x : ndarray
        nx1 array of dice roll value give fair or loaded
    y : ndarray
        (n+1)x1
    """
    x = np.ones(n)
    y = np.zeros(n)
    kLen, dLen = np.shape(emit_p)
    y[0] = rand.choice(kLen, p = start_p)
    x[0] = rand.choice(dLen, p = emit_p[y[0]])
    
    for k in range(1,n):
        y[k] = rand.choice(kLen, p = trans_p[y[k-1]])
        x[k] = rand.choice(dLen, p = emit_p[y[k]])
    
    return x,y
    
    
def viterbi(x, start_p, trans_p, emit_p):   
    xLen = len(x)
    kLen, dLen = np.shape(emit_p)
    
    # Initialize base cases (t == 0)
    v = np.zeros([xLen,kLen])
    v[0] = start_p[:]*emit_p[:,x[0]]
    # Initialize pointer
    lastMax = [np.argmax(start_p)]
    
    # Maximize V_l(i+1) = e_l(x(i+1))max_k a_kl V_k(i)
    for k in range(1,xLen):
        v[k] = emit_p[:,x[k]]*np.max(trans_p.dot(v[k-1]))
        lastMax.append(np.argmax(trans_p.dot(v[k-1])))
        
    
    return v, lastMax
    
def viterbi_wiki(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
    
    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]
    
    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        # Don't need to remember the old paths
        path = newpath
    
    # Return the most likely sequence over the given time frame
    n = len(obs) - 1
    
    (prob, state) = max((V[n][y], y) for y in states)
    
    return (prob, path[state])    