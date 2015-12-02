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
    c : float
        list of scalings of filter coefficients      
    """
    xLen = len(x)
    kLen, dLen = np.shape(emitP)
    
    # Initialize
    f = np.zeros([xLen,kLen])
    c = np.zeros(xLen)
    
    #base case
    f[0] = startP[:]*emitP[:,x[0]]
    c[0] = np.sum(f[0])
    f[0] *= 1/c[0]
    
        
    # Sum: f_l(i+1) = e_l(x(i+1))Sum_k a_kl f_k(i-1)
    for k in range(1,xLen):
        f[k] = emitP[:,x[k]]*np.dot(transP,f[k-1])
        c[k] = np.sum(f[k])
        f[k] *= 1/c[k]
        
    prob = np.sum(f[-1])
        
    return f, c, prob
    
    
def backward(x, startP, transP, emitP, cT = 1):
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
    c = np.ones(xLen)
    
    c[-1] = cT
    b[-1] *= 1/c[-1]
        
    # Maximize: V_l(i+1) = e_l(x(i+1))max_k a_kl V_k(i)
    for k in range(xLen-2,-1,-1):
        if k == 0:
            b[k] = np.dot(startP, emitP[:,x[k+1]]*b[k+1])
        else:
            b[k] = np.dot(transP,emitP[:,x[k+1]]*b[k+1])
        c[k] = np.sum(b[k])
        b[k] *= 1/c[k]
        
    # b_k(i) = sum_l (a_0l*e_l(x_1)*b_l(1)
    prob = np.sum(startP*emitP[:,x[0]]*b[0,:])
        
    return b, c, prob
    
    
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
    f, cF, probF = forward(x, startP, transP, emitP)
    b, cB, probB = backward(x, startP, transP, emitP, cF[-1])
    posterior = np.zeros([xLen, kLen])
    for k in range(xLen):
        posterior[k] = f[k]*b[k]
        
    return f, b, probF, probB, posterior
    
    
def baum_welch(x, startP, transP, emitP, delta, maxIt = 100):
    """
    Baum-Welch algorithm, derived from MIT open courseware pdf. See 
    "http://ocw.mit.edu/courses/aeronautics-and-astronautics/
    16-410-principles-of-autonomy-and-decision-making-fall-2010/lecture-notes/
    MIT16_410F10_lec21.pdf"
    
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
    startP : ndarray
        kx1 probabilities for starting in any hidden model state
    transP : ndarray
        kxk transition probabilities of hidden model, p_ij = P(i->j)
    emitP : ndarray
        kxd probability of emitting value from given hidden model state, k    
    """
    #Stopping condition parameters
    counter = 0
    converged = False
    
    #Scale parameters
    xLen = len(x)
    kLen,dLen = np.shape(emitP)
    
    #Initialize arrays for updating transition probabilites    
    transOut = np.zeros(kLen)
    transIJ = np.zeros([xLen, kLen, kLen])
    
    #Begin iteration
    while(not converged and counter < maxIt):
        converged = True
        counter += 1
        
        #emission probability indicator function
        indicator = np.zeros([kLen, dLen])

        #copies for tracking transition and emission probability changes
        transPOld = np.copy(transP)
        emitPOld = np.copy(emitP)
        
        #Estimate probability of observed and hidden states
        f, b, probF, probB, gamma = frwd_bkwd(x, startP, transP, emitP)

        #Additional values for updating transition and emission probabilities
        transOut = np.sum(gamma[:-1],1)
        emitOut = np.sum(gamma,1)

        #Update start probabilities
        startP = gamma[0]
        
        for k in range(kLen):

            #fill indicator function
            for m in range(dLen):
                if x[k] == m:
                    indicator[k,m] = 1
                    
            #update emission probabilites
            emitP[k] = np.dot(gamma,indicator)/emitOut[:]
            
            #fill transition update values and update            
            for l in range(kLen):
                transIJ[:,k,l] = f[k]*b[l]*transPOld[k,l]*emitPOld[:,x[l]]
                transP[k,l] = np.sum(transIJ[k,l])/transOut[k]
        
        #Check convergence of emission probabilities        
        if np.sum((emitP-emitPOld)**2)>delta:
            converged = converged and False
            
        #Check convergence of transition probabilities
        if np.sum((transP-transPOld)**2)>delta:
            converged = converged and False
        
    return startP, transP, emitP
    
    