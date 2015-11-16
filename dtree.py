#!/usr/bin/python


import numpy as np

def entropy_calc(y,x,x_classes):
    """
    Given an output set y and input variable x, calculate the entropy H(Y|X)
    
    Inputs
    ------
    y : array
        Output set y (1-D)
    x : array
	Input variable x (1-D), OR AN ARRAY OF ALL 0's
    x_classes : list
	The classes of x
    
    Returns
    -------
    ent : float
	The entropy H(Y|X) if X is non-trivial
	The entropy H(Y) if X is trivial

    Sample Call:
    ent = entropy_calc(map(int,y_pred),[0],[]) for trivial case
    ent = entropy_calc(map(int,y_pred),x_prob,all_dicts[1].keys()) for non-trivial case
    """

# This function assumes that Y is only two classes, 1 and 0
    if(sum(x) == 0): # X is trivial, just calculate H(Y)
        P1 = sum(y)/float(len(y))
        P0 = 1. - P1

        ent = -(P1*np.log(P1) + P0*np.log(P0))/np.log(2)
    else: # X is non-trivial, calculate the conditional probability
        npts = len(x)
        nclasses = len(x_classes)

        ent = 0.0

# CALCULATE THE ENTROPY
        for cl in range(nclasses):
            curr_class = x_classes[cl]
            cl_inds = np.where(x == curr_class)
            cl_size = np.size(cl_inds)

            if(cl_size == 0):
                continue
            else:

                y_vals = [y[j] for j in cl_inds[0]]
                P_x = np.size(cl_inds)/float(npts)


                P_x_1 = sum(y_vals)/float(np.size(cl_inds))
                P_x_0 = 1. - P_x_1

# If we get a nan, then either P_x_1 or P_x_0 = 0, so this should be mapped to 0
# The nan_to_num function does this
                ent = ent - P_x*np.nan_to_num(P_x_1*np.log(P_x_1) + P_x_0*np.log(P_x_0))/np.log(2)

    return ent


def I_Gain(dataset, keys):
    """
    Given a data set, define an output set y, input variables x, and calculate the information gain by splitting on this data set H(Y|X)
    
    Inputs
    ------
    dataset : dictionary
        Our dataset
    keys : list
        Keys for our dataset    
    """

# Some variables we could split on:
#	Anon Student ID
#	Problem Unit
#	Problem Section
#	Problem View
#	Problem Name
#	Step Start Time (normalized by FIRST student transaction time)
#	NOT CURRENTLY AVAILABLE: Step Start Time (normalized by FIRST student transaction time for each skill)
#	Step Duration
#	Incorrects
#	Hints
#	Skill
#	Opportunity

    y_pred = xy_train['Correct First Attempt']

    ent_base = dt.entropy_calc(y_pred,[0],[])

    opp_sqz = np.zeros((nsize),int)
    for i in range(nsize):
#    hh = [x for x in opp_digit[i,:] if x > 1]
        if sum(opp_array[i,:]) > 0.0:
            hh = [x for x in opp_array[i,:] if x > 0]
            opp_sqz[i] = np.min(hh)

    opp_bins = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    ob_sparse = [1,5,10,15]


    o_d = np.digitize(opp_sqz,opp_bins)
    o_ds = np.digitize(opp_sqz,ob_sparse)

#    ent1 = dt.entropy_calc(y_pred,opp_sqz,[0,1,5,10,15,20,25,30,35,40,45,50])


    stepdur_bins = [0,15,30,45,60]
    sd_d = np.digitize(xy_train['Step Duration (sec)'],stepdur_bins)




    p_strings = ['Step Duration','Opportunity']
    pp = [sd_d,o_ds]
    pp_bins = [[i for i in range(1,len(stepdur_bins)+1)],[i for i in range(1,len(ob_sparse)+1)]]
    b_remain = [0,1]

    branches = []
    b_inds = []

    y_lev = y_pred

    eb = [0.0,0.0]

    pred_array = np.zeros((4,5),int)

    if(branches == []):
        for i in range(len(pp)):
#        ent_od = dt.entropy_calc(y_pred,o_d,[i for i in range(1,len(opp_bins)+1)])
            eb[i] = dt.entropy_calc(y_pred,pp[i],pp_bins[i])
            IG[i] = ent_base - eb[i]
        branch_point = IG.index(np.max(IG))
        branches.append(p_strings[branch_point])
        b_inds.append(branch_point)
        del b_remain[branch_point]
        ent_base = min(eb)

    lev_classes = pp_bins[b_inds[0]]

    for lc in range(0,len(lev_classes)+1):
        loo = [kk for kk in np.where(pp[b_inds[0]] == lc)][0]
        y_lev = [y_pred[kk] for kk in loo]
        x_lev = [pp[b_inds[0]][kk] for kk in loo]

        csize = len(y_lev)
        print 'Big class size is '+ str(csize)

        d2class = pp_bins[b_remain[0]]
#        delirious = [kk for kk in np.where(pp[b_remain[0]] == lc)][0]
#        print delirious
        for lccc in range(1,len(d2class)+1):
            c2_array = [kk for kk in np.where(pp[b_remain[0]][loo] == lccc)][0]
#            print c2_array
            y_2class = [y_lev[kk] for kk in c2_array]
            c2size = len(y_2class)
#            c2size = len(c2_array)
            class_y = sum(y_2class)/float(sum(y_lev))
            print class_y
            pred_array[lc-1,lccc-1] = np.round(class_y)
#            print 'Small class size is '+ str(c2size)

b_inds = [1,0]
y_out = np.zeros((nsize),int)

for k in range(nsize):
    bin0 = pp[b_inds[0]][k]
    bin1 = pp[b_inds[1]][k]

    y_out[k] = pred_array[bin0-1][bin1-1]
    print y_out[k]

np.square((y_pred - y_out)).sum()/float(nsize)


#        print y_lev

#        print dt.entropy_calc(np.array(y_lev),np.array(x_lev),pp_bins[b_remain[0]])
#        print csize/float(nsize)*dt.entropy_calc(np.array(y_lev),np.array(x_lev),pp_bins[b_remain[0]])

# Fix this later
        ent_lev = ent_lev + csize/float(nsize)*dt.entropy_calc(np.array(y_lev),np.array(x_lev),pp_bins[b_remain[0]])


#        else:
#            lev_classes = pp_bins[b_inds[0]]
#            for lc in range(len(lev_classes)):
#                loo = [kk for kk in np.where(pp[b_inds[0]] == lc)][0]
#                y_lev = [y_pred[kk] for kk in loo]
#                x_lev = [pp[b_remain[0]][kk] for kk in loo]
