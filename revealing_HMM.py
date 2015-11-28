# Unknown, getting there, know

pi = np.array([0.31,0.34,0.35])

A = np.array([[0.32,0.34,0.34],[0.32,0.34,0.34],[0.32,0.34,0.34]])
B = np.array([[.48,.52],[.48,.52],[.48,.52]])

maxIters = 100
iters = 0
oldLogProb =-10000000000000000.

s = 175
y_output = y_stud[s][:inter_len[s]]


out_len = len(y_output)
states = np.shape(A)[0]

c_s = np.zeros((out_len),float)

alph = np.zeros((out_len,states),float)
bet = np.zeros((out_len,states),float)


while iters <= maxIters:

#Alpha pass
    for i in range(states):
        alph[0,i] = pi[i]*B[i][y_output[0]]
        c_s[0] = c_s[0] + alph[0,i]

    alph[0,:] = alph[0,:]/c_s[0]

    for t in range(out_len):
        for i in range(states):
            for j in range(states):
                alph[t,i] = alph[t,i] + alph[t-1,j]*A[j,i]
            alph[t,i] = alph[t,i]*B[i][y_output[t]]
            c_s[t] = c_s[t] + alph[t,i]

        alph[t,:] = alph[t,:]/c_s[t]

#Beta pass
    bet[out_len-1,:] = c_s[out_len-1]

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
