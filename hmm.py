# -*- coding: utf-8 -*-
"""
Created on Mon Nov 09 19:20:50 2015

@author: John
"""

import numpy as np
import numpy.random as rand
import operator

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
        kxk transition probabilities of hidden model, p_ij = P(i->j)
    emit_p : ndarray
        kxd probability of emitting value from given hidden model state, k
        
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
    """
    Dynamic programming algorithm for predicting hidden model state given the 
    observed emitted data, the transition probabilities between hidden model
    states and the emission probabilities for observations from each state.
    
    Inputs
    ------
    x : ndarray
        nx1 observed emitted data
    start_p : ndarray
        kx1 probabilities for starting in any hidden model state
    trans_p : ndarray
        kxk transition probabilities of hidden model, p_ij = P(i->j)
    emit_p : ndarray
        kxd probability of emitting value from given hidden model state, k
    
    Returns
    -------
    v : ndarray
        nxk probabilites of being in given k state at each step
    path : list
        nx1 path of hidden model state      
    """
    xLen = len(x)
    kLen, dLen = np.shape(emit_p)
    
    # Initialize base cases (t == 0)
    v = np.zeros([xLen,kLen])
    v[0] = start_p[:]*emit_p[:,x[0]]
    v2 = np.zeros([xLen,kLen])
    v2[0] = start_p[:]*emit_p[:,x[0]]
    # Initialize pointer
    path = [np.argmax(start_p)]
    
    # Maximize V_l(i+1) = e_l(x(i+1))max_k a_kl V_k(i)
    for k in range(1,xLen):
        v[k] = emit_p[:,x[k]]*np.max(trans_p.T.dot(v[k-1]))
        v2[k] = emit_p[:,x[k]]*np.max(trans_p.dot(v[k-1]))
        path.append(np.argmax(trans_p.T.dot(v[k-1])))
        
    
    return v, path, v2
    
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
            (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * 
                                emit_p[y][obs[t]], y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        # Don't need to remember the old paths
        path = newpath
        
    
    # Return the most likely sequence over the given time frame
    n = len(obs) - 1
    
    (prob, state) = max((V[n][y], y) for y in states)
    return V,path[state]
    #return (prob, path[state])
    
def viterbi_SE(observations,start_p,trans_p,emit_p):
    """Return the best path, given an HMM model and a sequence of observations"""
    # A - initialise stuff
    nSamples = len(observations)
    nStates = trans_p.shape[0] # number of states
    c = np.zeros(nSamples) #scale factors (necessary to prevent underflow)
    viterbi = np.zeros((nStates,nSamples)) # initialise viterbi table
    psi = np.zeros((nStates,nSamples)) # initialise the best path table
    best_path = np.zeros(nSamples); # this will be your output

    # B- appoint initial values for viterbi and best path (bp) tables - Eq (32a-32b)
    viterbi[:,0] = start_p.T * emit_p[:,observations[0]]
    c[0] = 1.0/np.sum(viterbi[:,0])
    #viterbi[:,0] = c[0] * viterbi[:,0] # apply the scaling factor
    psi[0] = 0;

    # C- Do the iterations for viterbi and psi for time>0 until T
    for t in range(1,nSamples): # loop through time
        for s in range (0,nStates): # loop through the states @(t-1)
            trans = viterbi[:,t-1] * trans_p[:,s]
            psi[s,t], viterbi[s,t] = max(enumerate(trans), key=operator.itemgetter(1))
            viterbi[s,t] = viterbi[s,t]*emit_p[s,observations[t]]

        c[t] = 1.0/np.sum(viterbi[:,t]) # scaling factor
        #viterbi[:,t] = c[t] * viterbi[:,t]

    # D - Back-tracking
    best_path[nSamples-1] =  viterbi[:,nSamples-1].argmax() # last state
    for t in range(nSamples-1,0,-1): # states of (last-1)th to 0th time step
        best_path[t-1] = psi[best_path[t],t]

    return best_path, viterbi
    
def fwd_bkw_wiki(x, states, a_0, a, e, end_st):
    L = len(x)
 
    fwd = []
    f_prev = {}
    # forward part of the algorithm
    for i, x_i in enumerate(x):
        f_curr = {}
        for st in states:
            if i == 0:
                # base case for the forward part
                prev_f_sum = a_0[st]
            else:
                prev_f_sum = sum(f_prev[k]*a[k][st] for k in states)
 
            f_curr[st] = e[st][x_i] * prev_f_sum
 
        fwd.append(f_curr)
        f_prev = f_curr
 
    p_fwd = sum(f_curr[k]*a[k][end_st] for k in states)
 
    bkw = []
    b_prev = {}
    # backward part of the algorithm
    for i, x_i_plus in enumerate(reversed(x[1:]+(None,))):
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = a[st][end_st]
            else:
                b_curr[st] = sum(a[st][l]*e[l][x_i_plus]*b_prev[l] for l in states)
 
        bkw.insert(0,b_curr)
        b_prev = b_curr
 
    p_bkw = sum(a_0[l] * e[l][x[0]] * b_curr[l] for l in states)
 
    # merging the two parts
    posterior = []
    for i in range(L):
        posterior.append({st: fwd[i][st]*bkw[i][st]/p_fwd for st in states})
 
    assert p_fwd == p_bkw
    return fwd, bkw, posterior