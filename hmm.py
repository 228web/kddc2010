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
        
def general_hmm_dat(n,startP, transP, emitP):
    """
    Loaded vs. unloaded die example
    
    Inputs
    ------
    n : int
        number of points to make
    startP : ndarray
        kx1 probabilities for starting in any hidden model state
    transP : ndarray
        kxk transition probabilities of hidden model, p_ij = P(i->j)
    emitP : ndarray
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
    kLen, dLen = np.shape(emitP)
    y[0] = rand.choice(kLen, p = startP)
    x[0] = rand.choice(dLen, p = emitP[y[0]])
    
    for k in range(1,n):
        y[k] = rand.choice(kLen, p = transP[y[k-1]])
        x[k] = rand.choice(dLen, p = emitP[y[k]])
    
    return x,y
    
    
def viterbi(x, startP, transP, emitP):
    """
    Dynamic programming algorithm for predicting hidden model state given the 
    observed emitted data, the transition probabilities between hidden model
    states and the emission probabilities for observations from each state.
    
    Inputs
    ------
    x : ndarray
        nx1 observed emitted data
    startP : ndarray
        kx1 probabilities for starting in any hidden model state
    transP : ndarray
        kxk transition probabilities of hidden model, p_ij = P(i->j)
    emitP : ndarray
        kxd probability of emitting value from given hidden model state, k
    
    Returns
    -------
    v : ndarray
        nxk probabilites of being in given k state at each step
    path : ndarray
        nx1 path of hidden model state      
    """
    xLen = len(x)
    kLen, dLen = np.shape(emitP)
    
    # Initialize base cases (t == 0)
    v = np.zeros([xLen,kLen])
    v[0] = startP[:]*emitP[:,x[0]]
    # Initialize path table and best path
    pointer = np.zeros([xLen,kLen])
    path = np.zeros(xLen)
        
    # Maximize: V_l(i+1) = e_l(x(i+1))max_k a_kl V_k(i)
    for k in range(1,xLen):
        for l in range(kLen):
            trans = transP[l]*v[k-1]
            v[k,l] = emitP[l,x[k]]*np.max(trans)
            pointer[k,l] = np.argmax(trans)
          
    # Set the final path state as max V_k
    path[-1] = np.argmax(v[-1])
    # Step backward along path
    for k in range(xLen-1,0,-1):
        path[k-1] = pointer[k,path[k]]
        
    return v, path

def forward(x, startP, transP, emitP):
    """
    Dynamic programming algorithm for predicting probability of observed 
    emitted data series, aka filtering.
    
    Inputs
    ------
    x : ndarray
        nx1 observed emitted data
    startP : ndarray
        kx1 probabilities for starting in any hidden model state
    transP : ndarray
        kxk transition probabilities of hidden model, p_ij = P(i->j)
    emitP : ndarray
        kxd probability of emitting value from given hidden model state, k
    
    Returns
    -------
    f : ndarray
        forward probabilities at each series point x[k]
    prob : float
        probability of series of emitted data x      
    """
    xLen = len(x)
    kLen, dLen = np.shape(emitP)
    
    # Initialize base cases (t == 0)
    f = np.zeros([xLen,kLen])
    f[0] = startP[:]*emitP[:,x[0]]
        
    # Sum: f_l(i+1) = e_l(x(i+1))Sum_k a_kl f_k(i-1)
    for k in range(1,xLen):
        f[k] = emitP[:,x[k]]*np.dot(transP,f[k-1])
        
    prob = np.sum(f[-1])
        
    return f, prob
    
def backward(x, startP, transP, emitP):
    """
    Dynamic programming algorithm for predicting probability of observed 
    emitted data series, given hidden model state.
    
    Inputs
    ------
    x : ndarray
        nx1 observed emitted data
    startP : ndarray
        kx1 probabilities for starting in any hidden model state
    transP : ndarray
        kxk transition probabilities of hidden model, p_ij = P(i->j)
    emitP : ndarray
        kxd probability of emitting value from given hidden model state, k
    
    Returns
    -------
    b : ndarray
        forward probabilities at each series point x[k]
    prob : float
        probability of series of emitted data x 
    """
    xLen = len(x)
    kLen, dLen = np.shape(emitP)
    
    # Initialize base cases (t == 0)
    b = np.ones([xLen,kLen])
        
    # Maximize: V_l(i+1) = e_l(x(i+1))max_k a_kl V_k(i)
    for k in range(xLen-2,-1,-1):
        if k == 0:
            b[k] = np.dot(startP, emitP[:,x[k+1]]*b[k+1])
        else:
            b[k] = np.dot(transP,emitP[:,x[k+1]]*b[k+1])
        
    # b_k(i) = sum_l (a_0l*e_l(x_1)*b_l(1)
    prob = np.sum(startP*emitP[:,x[0]]*b[0,:])
        
    return b, prob
    
def frwd_bkwd(x, startP, transP, emitP):
    """
    Computes posterior probability of hidden state given observation data,
    aka smoothing.
    
    Inputs
    ------
    x : ndarray
        nx1 observed emitted data
    startP : ndarray
        kx1 probabilities for starting in any hidden model state
    transP : ndarray
        kxk transition probabilities of hidden model, p_ij = P(i->j)
    emitP : ndarray
        kxd probability of emitting value from given hidden model state, k
    
    Returns
    -------
    f : ndarray
        forward probabilities at each series point x[k]
    probF : float
        probability of series of emitted data x
    b : ndarray
        backward probabilities at each series point x[k]
    probB : float
        probability of series of emitted data x
    posterior : ndarray
        posterior probability P(pi_i = k|x)    
    """
    
    xLen = len(x)
    kLen, dLen = np.shape(emitP)
    f, probF = forward(x, startP, transP, emitP)
    b, probB = backward(x, startP, transP, emitP)
    posterior = np.zeros([xLen, kLen])
    for k in range(xLen):
        posterior[k] = f[k]*b[k]/probF
        
    return f, b, probF, probB, posterior
    
def baum_welch(x, startP, transP, emitP, delta, maxIt = 100):
    #Wrong!
    counter = 0
    converged = False
    xLen = len(x)
    kLen = len(startP)
    transOut = np.zeros(kLen)
    transIJ = np.zeros([kLen,xLen])
    while(not converged and counter < maxIt):
        converged = True
        counter += 1
        f, b, probF, probB, posterior = frwd_bkwd(x, startP, transP, emitP)
        transOut = np.sum(posterior[:-1],1)
        transPOld = np.copy(transP)
        
        emitPOld = np.copy(emitP)
        for k in range(kLen):
            emitInd = np.zeros([kLen,xLen])
            transIJ[k] = f[k]*b*transPOld[k]*emitP[:,x[k]]/probF
            transP[k] = np.sum(transIJ)/np.sum(transOut)
            if abs(transP[k]-transPOld[k])>delta:
                converged = converged and False
            for l in range(xLen):
                if x[l] == k:
                    emitInd += 1
            emitP[k] = np.sum(posterior*emitInd)/np.sum(posterior,1)
        
    return transP, emitP
    
def viterbi_wiki(obs, states, startP, transP, emitP):
    V = [{}]
    path = {}
    
    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = startP[y] * emitP[y][obs[0]]
        path[y] = [y]
    
    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max((V[t-1][y0] * transP[y0][y] * 
                                emitP[y][obs[t]], y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        # Don't need to remember the old paths
        path = newpath
        
    
    # Return the most likely sequence over the given time frame
    n = len(obs) - 1
    
    (prob, state) = max((V[n][y], y) for y in states)
    return V,path[state]
    #return (prob, path[state])
    
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