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
                P_x = cl_size/float(npts)


                P_x_1 = sum(y_vals)/float(cl_size)
                P_x_0 = 1. - P_x_1

# If we get a nan, then either P_x_1 or P_x_0 = 0, so this should be mapped to 0
# The nan_to_num function does this
                ent = ent - P_x*np.nan_to_num(P_x_1*np.log(P_x_1) + P_x_0*np.log(P_x_0))/np.log(2)

    return ent

def calc_dtree(y_vals,dataset,dataset_bins,var_strings,tree_order):
    """
    Given a data set, and the strings of a dictionary, return a decision tree with the order given in tree order
    
    Inputs
    ------
    y_vals : array
        The dependent variable
    dataset : list
        The dataset we wish to use to construct our decision tree
    dataset_bins : list
        The bins for the variables inside of the dataset
    var_strings : list
        Strings corresponding to the dataset
    tree_order : array
        Array of indices corresponding to levels of the branches of our tree

   Returns
    -------
    branch_indices : list
        A k-dimensional list (k is length of tree order) with the global indices corresponding to the given bins.

    Sample Call:
        final_branch = calc_dtree(y_pred,dataset,dataset_bins,var_strings,tree_order)
    """

    depth_tree = len(var_strings)
    branch_indices = []
    shape_array = []

# While the tree is hard coded in this function,
# segments of this code are kept general to make
# adaptive trees easier to implement
    for d in range(depth_tree):
        b_index = tree_order[d]
        lev_classes = dataset_bins[b_index]
        num_branches = len(lev_classes)

# Prepare pred_array size for populating
        if d == 0: # Create tree root
            shape_array = np.array([num_branches])
            curr_branch = branch_fill_func(dataset[b_index],lev_classes)
            branch_indices = curr_branch
        else: # Create branches
            shape_array = np.append(shape_array,num_branches)
#            print 'Branch indices shape is ' + str(np.shape(branch_indices))
#            print branch_indices
            branch_indices = branch_recurse(dataset[b_index],lev_classes,branch_indices,d+1,0)

    return branch_indices

def calc_adapt_dtree(y_vals,dataset,dataset_bins,var_strings):
    """
    Given a data set and a set of variables, calculate an adaptive decision tree
    
    Inputs
    ------
    y_vals : array
        The dependent variable
    dataset : list
        The dataset we wish to use to construct our decision tree
    dataset_bins : list
        The bins for the variables inside of the dataset
    var_strings : list
        Strings corresponding to the dataset

   Returns
    -------
    branch_indices : list
        A k-dimensional list (k is length of var_strings) with the global indices corresponding to the given bins.
    tree_order : array
        Array of indices corresponding to the levels of the tree

    Sample Call:
        [final_branch,tree_order] = calc_adapt_dtree(y_pred,dataset,dataset_bins,var_strings)

    """

    depth_tree = len(var_strings)
    branch_indices = []
    shape_array = []

    tree_order = np.zeros((depth_tree),int)
    tree_remain = [i for i in range(depth_tree)]

    ent_base = entropy_calc(y_vals,[0],[])

    for d in range(depth_tree):
#        b_index = tree_order[d]
#        lev_classes = dataset_bins[b_index]
#        num_branches = len(lev_classes)

        print 'This is tree remain: ' + str(tree_remain)
# Prepare pred_array size for populating
        if d == 0: # Create tree root
            IG = np.zeros((len(tree_remain)),float)
            IG[:] = ent_base
            for l in range(depth_tree):
                IG[l] = IG[l] - entropy_calc(y_vals,dataset[l],dataset_bins[l])
            root_point = np.where(IG == np.max(IG))[0]
            b_index = tree_remain[root_point]
            tree_order[d] = b_index
            del tree_remain[b_index]
            ent_base = ent_base - IG[root_point]

            lev_classes = dataset_bins[b_index]
            num_branches = len(lev_classes)
            shape_array = np.array([num_branches])

            branch_indices = branch_fill_func(dataset[b_index],lev_classes)
        else: # Create branches
            rem_depth = len(tree_remain)
            IG = np.zeros((rem_depth),float)
            IG[:] = ent_base
            for x in range(rem_depth):
                poss_index = tree_remain[x]
                [entropy_prov,class_size] = entropy_recurse(y_vals,dataset[poss_index],dataset_bins[poss_index],branch_indices,d,0)
                IG[x] = IG[x] - entropy_prov
            root_point = np.where(IG == np.max(IG))[0]
            b_index = tree_remain[root_point]
            tree_order[d] = b_index
            del tree_remain[root_point]
            ent_base = ent_base - IG[root_point]

            lev_classes = dataset_bins[b_index]
            num_branches = len(lev_classes)


#            print IG

            shape_array = np.append(shape_array,num_branches)
            branch_indices = branch_recurse(dataset[b_index],lev_classes,branch_indices,d+1,0)


    return [branch_indices,np.array(tree_order)]


# Functions used to calculate the decision tree
def branch_recurse(x_array,dataset_bins,b_i,depth,curr_depth):
    if depth - 2 > curr_depth:
        final_array = []
        curr_len = np.shape(b_i)[0]
#        print 'Current depth is ' + str(curr_depth)
#        print 'Expected loop is ' + str(curr_len)
        for i in range(curr_len):
#            print i
            final_array.append(branch_recurse(x_array,dataset_bins,b_i[i],depth,curr_depth + 1))
    else:
#         print 'Length: ' + str(x_array[300])

         prov_array = branch_fill_func(x_array,dataset_bins)
#         print np.shape(prov_array)
         final_array = [None]*len(b_i)
         for i in range(len(b_i)):
             final_array[i] = [None]*len(prov_array)
#	 print 'Final array, depth 1 shape is ' +str(np.shape(final_array))

         for root_i in range(len(b_i)):
             for prov_i in range(len(prov_array)):
                 final_array[root_i][prov_i] = np.intersect1d(prov_array[prov_i],b_i[root_i])
         print 'Done with recursion'
#	 print 'Final array, depth 1 shape is ' +str(np.shape(final_array))
    return final_array

def entropy_recurse(y_vals,x_vals,dataset_bins,b_i,depth,curr_depth):
    ent = 0.0
    class_size = 0
    if depth > curr_depth:
        curr_len = np.shape(b_i)[0]
        for i in range(curr_len):
            [ent_c,c_c] = entropy_recurse(y_vals,x_vals,dataset_bins,b_i[i],depth,curr_depth+1) 
            ent = ent + ent_c
            class_size = class_size + c_c
#            print class_size
        if class_size >0: #Else case unnecessary, this shouldn't contribute to entropy at all.
            ent = ent/class_size
    else:
        class_size = len(b_i)
        y_class = np.array([y_vals[kk] for kk in b_i])
        x_class = np.array([x_vals[kk] for kk in b_i])
        if len(y_class) > 0:
            ent = class_size*entropy_calc(y_class,x_class,dataset_bins)
        else:
            ent = 0.0
    return ent,class_size

def branch_fill_func(data,vals):
    b = len(vals)
    branch_fill = [None]*b
    for x in range(b):
        branch_fill[x] = [kk for kk in np.where(data == vals[x])][0]
    return branch_fill
