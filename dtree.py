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
    """

# This function assumes that Y is only two classes, 1 and 0
    if(sum(x) == 0): # X is trivial, just calculate H(Y)
        P1 = sum(y)/len(y)
        P0 = 1. - P1

        ent = -(P1*np.log(P1) + P0*np.log(P0))/np.log(2)
    else: # X is non-trivial, calculate the conditional probability
        npts = len(x)
        nclasses = len(x_classes)
# CALCULATE THE ENTROPY
        for cl in range(nclasses):
            curr_class = x_classes[cl]
            cl_inds = np.where(x == curr_class)
            in_size = len(cl_inds)



