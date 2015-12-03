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
    
    #Need transpose of transition probability matrix
    transPT = transP.T    
        
    # Sum: f_l(i+1) = e_l(x(i+1))Sum_k a_kl f_k(i-1)
    for k in range(1,xLen):
        f[k] = emitP[:,x[k]]*np.dot(transPT,f[k-1])
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
    cT : ndarray, optional
        This can either be taken from the scaled forward pass or regenerated
    
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
        posterior[k] *= 1/np.sum(posterior[k])
        
    return f, b, probF, probB, posterior
    
    
def baum_welch(x, startP, transP, emitP, delta, maxIt = 100):
    """
    Baum-Welch algorithm, derived from MIT open courseware pdf. See 
    "http://ocw.mit.edu/courses/aeronautics-and-astronautics/
    16-410-principles-of-autonomy-and-decision-making-fall-2010/lecture-notes/
    MIT16_410F10_lec21.pdf"
    
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
    maxIt : ndarray, optional
        maximum number of iterations through expectation maximization
    
    Returns
    -------
    startOut : ndarray
        kx1 probabilities for starting in any hidden model state
    transOut : ndarray
        kxk transition probabilities of hidden model, p_ij = P(i->j)
    emitOut : ndarray
        kxd probability of emitting value from given hidden model state, k    
    """
    #Stopping condition parameters
    counter = 0
    converged = False
    
    #Scale parameters
    xLen = len(x)
    kLen,dLen = np.shape(emitP)
    
    #Initialize output arrays
    startOut = np.zeros(kLen)
    emitOut = np.zeros([kLen,dLen])
    transOut = np.zeros([kLen,kLen])
    
    #Initialize arrays for updating transition probabilites    
    transDenom = np.zeros(kLen)
    transIJ = np.zeros([xLen-1, kLen, kLen])
    
    #Begin iteration
    while(not converged and counter < maxIt):
        converged = True
        counter += 1
        
        #emission probability indicator function
        indicator = np.zeros([xLen, dLen])

        #copies for tracking transition and emission probability changes
        transPOld = np.copy(transP)
        emitPOld = np.copy(emitP)
        
        #Estimate probability of observed and hidden states
        f, b, probF, probB, gamma = frwd_bkwd(x, startP, transP, emitP)

        #Additional values for updating transition and emission probabilities
        transDenom = np.sum(gamma[:-1],0)
        emitDenom = np.sum(gamma,0)

        #Update start probabilities
        startOut = gamma[0]
        
        #Build the indicator function
        for m in range(dLen):
            indicator[:,m] = np.array([1 if x[l]==m else 0 for l in range(xLen)])            
            
        for n in range(xLen-1):
            for k in range(kLen):
                #Fill in transition update values                   
                transIJ[n,k,:] = f[n,k]*b[n+1,:]*transPOld[k,:]*emitPOld[:,x[n+1]]
        
        for k in range(kLen):                  
            #update emission probabilities
            emitOut[k] = np.dot(gamma[:,k],indicator)/emitDenom[k]
            #update transition probabilities
            transOut[k] = np.sum(transIJ[:,k,:],0)/transDenom[k]
            #rescale transition probabilities since b scaling of probabilities 
            #not exactly matched to f scalings
            transOut[k] *= 1/np.sum(transP[k])
        
        #Check convergence of emission probabilities        
        if np.sum((emitP-emitPOld)**2)>delta:
            converged = converged and False
            
        if converged:
            print 'Converged! (in emission probs)'
            
        #Check convergence of transition probabilities
        if np.sum((transP-transPOld)**2)>delta:
            converged = converged and False
            
        if converged:
            print 'Converged!(in transition probs too!)'
        
    return startOut, transOut, emitOut
    
    
def baum_welch_case(x, startP, transP, emitP, splitIds):
    """
    Baum-Welch algorithm, derived from MIT open courseware pdf as above, but 
    applied on each iteration to a new segment of observation data. See 
    "http://ocw.mit.edu/courses/aeronautics-and-astronautics/
    16-410-principles-of-autonomy-and-decision-making-fall-2010/lecture-notes/
    MIT16_410F10_lec21.pdf"
    
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
    splitIds : ndarray
        array of indices over which to iterate baum welch on
    
    Returns
    -------
    startOut : ndarray
        kx1 probabilities for starting in any hidden model state
    transOut : ndarray
        kxk transition probabilities of hidden model, p_ij = P(i->j)
    emitOut : ndarray
        kxd probability of emitting value from given hidden model state, k    
    """    
    #Scale parameters
    kLen,dLen = np.shape(emitP)
    
    #Initialize output arrays
    startOut = np.copy(startP)
    transOut = np.copy(transP)
    emitOut = np.copy(emitP)
    
    #Initialize arrays for updating transition probabilites    
    transDenom = np.zeros(kLen)
    
    #Begin iteration
    for l in range(len(splitIds)+1):
        #x scale parameter changes each iteration
        #Case dependent since observations split into different segments
        if l == 0:
            xLen = len(x[:splitIds[l]])
        elif l == len(splitIds):
            xLen = len(x[splitIds[l-1]:])
        else:
            xLen = len(x[splitIds[l-1]:splitIds[l]])
        transIJ = np.zeros([xLen-1, kLen, kLen])

        #emission probability indicator function
        indicator = np.zeros([xLen, dLen])

        #copies for tracking transition and emission probability changes
        transPOld = np.copy(transOut)
        emitPOld = np.copy(emitOut)
        
        #Estimate probability of observed and hidden states
        #Case dependent since observations split into different segments
        if l == 0:
            f, b, probF, probB, gamma = frwd_bkwd(x[:splitIds[l]], 
                                                  startOut, transOut, emitOut)
        elif l == len(splitIds):
            f, b, probF, probB, gamma = frwd_bkwd(x[splitIds[l-1]:], 
                                                  startOut, transOut, emitOut)
        else:
            f, b, probF, probB, gamma = frwd_bkwd(x[splitIds[l-1]:splitIds[l]], 
                                                  startOut, transOut, emitOut)
        #Additional values for updating transition and emission probabilities
        transDenom = np.sum(gamma[:-1],0)
        emitDenom = np.sum(gamma,0)

        #track whether start probability shows a problem
        startPOld = np.copy(startOut)
        
        #Update start probabilities
        startOut = gamma[0]
        print startOut
        if np.isnan(np.sum(startOut)):
            return startPOld, transPOld, emitPOld
        
        #Build the indicator function
        for m in range(dLen):
            indicator[:,m] = np.array([1 if x[l]==m else 0 for l in range(xLen)])            
            
        for n in range(xLen-1):
            for k in range(kLen):
                #Fill in transition update values                   
                transIJ[n,k,:] = f[n,k]*b[n+1,:]*transPOld[k,:]*emitPOld[:,x[n+1]]
        
        for k in range(kLen):                  
            #update emission probabilities
            emitOut[k] = np.dot(gamma[:,k],indicator)/emitDenom[k]
            #update transition probabilities
            transOut[k] = np.sum(transIJ[:,k,:],0)/transDenom[k]
            #rescale transition probabilities since b scaling of probabilities 
            #not exactly matched to f scalings
            transOut[k] *= 1/np.sum(transOut[k])
        
    return startOut, transOut, emitOut
    

def rev_hmm_bw(y_output, pi, A, B,maxIters=1):
    out_len = len(y_output)
    states = np.shape(A)[0]
    iters = 0
    
    c_s = np.zeros((out_len),float)
    
    alph = np.zeros((out_len,states),float)
    bet = np.zeros((out_len,states),float)
    
    
    while iters <= maxIters:
    
        #Alpha pass
        for i in range(states):
            alph[0,i] = pi[i]*B[i][y_output[0]]
            c_s[0] = c_s[0] + alph[0,i]
    
        alph[0,:] = alph[0,:]/c_s[0]
    
        for t in range(1,out_len):
            for i in range(states):
                for j in range(states):
                    alph[t,i] = alph[t,i] + alph[t-1,j]*A[j,i]
                alph[t,i] = alph[t,i]*B[i][y_output[t]]
                c_s[t] = c_s[t] + alph[t,i]
    
            alph[t,:] = alph[t,:]/c_s[t]
    
        #Beta pass
        bet[out_len-1,:] = 1/c_s[out_len-1]
    
        for t in range(out_len-2,-1,-1):
            for i in range(states):
                for j in range(states):
                    bet[t,i] = bet[t,i] + A[i,j]*B[j][y_output[t+1]]*bet[t+1,j]
                bet[t,:] = bet[t,:]/c_s[t]
    
    
        #Estimate gamma
        gamma_2 = np.zeros((out_len,states,states),float)
        gamma_1 = np.zeros((out_len,states),float)
    
        for t in range(out_len-1):
            denom = 0.0
    
            for i in range(states):
                for j in range(states):
                    denom = denom + alph[t,i]*A[i,j]*B[j][y_output[t+1]]*bet[t+1,j]
            for i in range(states):
                for j in range(states):
                    gamma_2[t,i,j] = alph[t,i]*A[i,j]*B[j][y_output[t+1]]*bet[t+1,j]/denom
                    gamma_1[t,i] = gamma_1[t,i] + gamma_2[t,i,j]
    
        t = out_len-1
        denom = 0.0
        for i in range(states):
            denom = denom + alph[t,i]
        for i in range(states):
            gamma_1[t,i] = alph[t,i]/denom
    
        #Re-estimate A,B,pi
        for i in range(states):
            pi[i] = gamma_1[0][i]
    
        for i in range(states):
            for j in range(states):
                numer = 0.0
                denom = 0.0
                for t in range(0,out_len - 1):
                    numer = numer + gamma_2[t,i,j]
                    denom = denom + gamma_1[t,i]
                A[i,j] = numer/denom
    
        for i in range(states):
            for j in range(2):
                numer = 0.0
                denom = 0.0
                for t in range(0,out_len):
                    if(y_output[t] == j):
                        numer = numer + gamma_1[t,i]
                    denom = denom + gamma_1[t,i]
                B[i,j] = numer/denom
    
    
        # Compute log prob
        LogProb = 0
        for i in range(out_len):
            LogProb = LogProb + np.log(c_s[t])
        LogProb = -LogProb
    
        print str(iters) + ' : ' + str(LogProb)
        iters = iters + 1
        
    return alph, bet, c_s, pi, A, B