
import numpy as np

def forward_scaled(x,startP,transP,emitP):
    xLen = len(x)
    kLen = np.shape(transP)[0]

    f = np.zeros((xLen,kLen),float)
    scal = np.zeros((xLen),float)

    f[0,:] = startP[:]*emitP[:,x[0]]
    scal[0] = scal[0] + np.sum(f[0])

    f[0,:] = f[0,:]/scal[0]

    for t in range(1,xLen):
        f[t,:] = f[t,:] + np.dot(f[t-1],transP)
        f[t,:] = f[t,:]*emitP[:,x[t]]
        scal[t] = scal[t] + np.sum(f[t,:])

        f[t,:] = f[t,:]/scal[t]

    return f,scal

def backward_scaled(x,startP,transP,emitP,scaling):
    xLen = len(x)
    kLen = np.shape(transP)[0]

    b = np.zeros((xLen,kLen),float)

    b[:,:] = 0.0
    b[-1,:] = 1./scaling[-1]

    for t in range(xLen-2,0,-1):
        for i in range(kLen):
            for j in range(kLen):
                b[t,i] = b[t,i] + transP[i,j]*emitP[j,x[t+1]]*b[t+1,j]
        b[t,:] = b[t,:]/scaling[t]

    for i in range(kLen):
        for j in range(kLen):
            b[0,i] = b[0,i] + transP[i,j]*emitP[j,x[1]]*b[1,j]
    b[0,:] = b[0,:]/scaling[0]

    return b

def gam_calc(x,startP,transP,emitP,f,b):
    xLen = len(x)
    kLen = np.shape(transP)[0]


    g2 = np.zeros((xLen,kLen,kLen),float)
    g1 = np.zeros((xLen,kLen),float)

    for t in range(xLen-1):
        denom = 0.0

        for i in range(kLen):
            for j in range(kLen):
                denom = denom + f[t,i]*transP[i,j]*emitP[j,x[t+1]]*b[t+1,j]
        for i in range(states):
            for j in range(states):
                g2[t,i,j] = f[t,i]*transP[i,j]*emitP[j,x[t+1]]*b[t+1,j]/denom
                g1[t,i] = g1[t,i] + g2[t,i,j]

    t = xLen-1
    denom = 0.0
    denom = denom + np.sum(f[t])
    g1[t,:] = f[t,:]/denom


    return g1,g2


def state_recalc(x,startP,transP,emitP,g1,g2):
    xLen = len(x)
    kLen = np.shape(transP)[0]
    dLen = np.shape(emitP)[1]

    startP = g1[0]

    for i in range(kLen):
        for j in range(kLen):
            numer = 0.0
            denom = 0.0
            for t in range(0,xLen - 1):
                numer = numer + g2[t,i,j]
                denom = denom + g1[t,i]
            transP[i,j] = numer/denom

    for i in range(kLen):
        for j in range(dLen):
            numer = 0.0
            denom = 0.0
            for t in range(0,xLen):
                if(x[t] == j):
                    numer = numer + g1[t,i]
                denom = denom + g1[t,i]
            emitP[i,j] = numer/denom

    return startP,transP,emitP

# Unknown, getting there, know

pi = np.array([0.31,0.34,0.35])

A = np.array([[0.32,0.35,0.33],[0.32,0.35,0.33],[0.32,0.35,0.33]])
B = np.array([[.48,.52],[.48,.52],[.48,.52]])

maxIters = 100
iters = 0
oldLogProb =-10000000000000000.

s = 1
y_output = y_stud[s][:inter_len[s]]


out_len = len(y_output)
states = np.shape(A)[0]


while iters <= maxIters:
#Alpha pass
    alph,c_s = forward_scaled(y_output,pi,A,B)



#Beta pass
    bet = backward_scaled(y_output,pi,A,B,c_s)

#Estimate gamma
    gamma_1,gamma_2 = gam_calc(y_output,pi,A,B,alph,bet)

#Re-estimate A,B,pi
    piNew, Anew,Bnew = state_recalc(y_output,pi,A,B,gamma_1,gamma_2)


    pi = piNew
    A = Anew
    B = Bnew

# Compute log prob
    LogProb = 0
    for i in range(out_len):
        LogProb = LogProb + np.log(1./c_s[t])
    LogProb = -LogProb

    print str(iters) + ' : ' + str(LogProb)
    iters = iters + 1
