#!/usr/bin/python


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

                ent = ent - P_x*np.nan_to_num(P_x_1*np.log(P_x_1) + P_x_0*np.log(P_x_0))/np.log(2)

    return ent
