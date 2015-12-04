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

    Sample Call
    -----------
    ent = entropy_calc(map(int,y_pred),[0],[]) for trivial case
    ent = entropy_calc(map(int,y_pred),x_prob,all_dicts[1].keys()) for the Snon-
        trivial case
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
# The probability we are in the current class
                P_x = cl_size/float(npts)

# The y outputs in this class
                P_x_1 = sum(y_vals)/float(cl_size)
                P_x_0 = 1. - P_x_1

# If we get a nan, then either P_x_1 or P_x_0 = 0, so this should be mapped to 0
# The nan_to_num function does this
                ent = ent - P_x*np.nan_to_num(P_x_1*np.log(P_x_1) + P_x_0*np.log(P_x_0))/np.log(2)

    return ent

def calc_dtree(y_vals,dataset,dataset_bins,var_strings,tree_order):
    """
    Given a data set, and the strings of a dictionary, return a decision tree 
    with the order given in tree order
    
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
        A k-dimensional list (k is length of tree order) with the global 
        indices corresponding to the given bins.
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
        # Create tree root
        if d == 0: 
            shape_array = np.array([num_branches])
            curr_branch = branch_fill_func(dataset[b_index],lev_classes)
            branch_indices = curr_branch
        # Create branches
        else: 
            shape_array = np.append(shape_array,num_branches)
#            print 'Branch indices shape is ' + str(np.shape(branch_indices))
#            print branch_indices
            branch_indices = branch_recurse(dataset[b_index],lev_classes,
                                            branch_indices,d+1,0)

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

    Sample Call
    -----------
    final_branch,tree_order = calc_adapt_dtree(y_pred,dataset,dataset_bins,var_strings)

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
        # Create tree root
        if d == 0: 
            #initialize array of information gain
            IG = np.ones((len(tree_remain)),float)
            IG *= ent_base
            
            # Calculate entropy for each candidate independent variable
            for l in range(depth_tree): 
                IG[l] = IG[l] - entropy_calc(y_vals,dataset[l],dataset_bins[l])
                
            # Root is location split with maximum information gain
            root_point = np.argmax(IG)
            b_index = tree_remain[root_point] # Find the best one
            tree_order[d] = b_index
            
            # Discard split from further use
            del tree_remain[b_index]
            
            # Update the current entropy
            ent_base = ent_base - IG[root_point]

            lev_classes = dataset_bins[b_index]
            num_branches = len(lev_classes)
            shape_array = np.array([num_branches])

            branch_indices = branch_fill_func(dataset[b_index],lev_classes)
        # Create branches
        else: 
            rem_depth = len(tree_remain)
            
            #initialize array of new information gain for remaining variables
            IG = np.zeros((rem_depth),float)
            IG[:] = ent_base
            
            # Calculate the entropy for each candidate independent variable
            for x in range(rem_depth): 
                poss_index = tree_remain[x]
                [entropy_prov,class_size] = entropy_recurse(y_vals,dataset[poss_index],dataset_bins[poss_index],branch_indices,d,0)
                IG[x] = IG[x] - entropy_prov
                
            # New split is one with maximum information gain
            root_point = np.where(IG == np.max(IG))[0]
            b_index = tree_remain[root_point] # Find the best one
            tree_order[d] = b_index
            
            # Discard split from further use
            del tree_remain[root_point]
            
            # Update the current entropy
            ent_base = ent_base - IG[root_point]

            lev_classes = dataset_bins[b_index]
            num_branches = len(lev_classes)


            shape_array = np.append(shape_array,num_branches)
            branch_indices = branch_recurse(dataset[b_index],lev_classes,branch_indices,d+1,0)


    return [branch_indices,np.array(tree_order)]


# Functions used to calculate the decision tree
def branch_recurse(x_array,dataset_bins,b_i,depth,curr_depth):
    """
    Recursively create branches for a tree
    
    Inputs
    ------
    x_array : array
        The independent variable for which we want to create branches
    dataset_bins : list
        The classes of x_array
    b_i : list
        A list with the same shape as the tree that contains the indices for each class
        First dimension - the indices for each of the different classes of tree root x_1 = i b_i[0] = all indices where x_1 = 0...
        Second dimension - subclass: b_i[0,0] = all indices where x_1 = 0, x_2 = 0, b_i[3,4] = all indices where x_1 = 3, x_2 = 4
        Third dimension - etc... (b_i[2,5,3] = all indices where x_1 = 2, x_2 = 5, x_3 = 3...
    depth : integer
        Desired depth of full tree
    curr_depth : integer
        Current depth of tree

    Returns
    -------
    final_array : list
        Gives b_i for a tree with a depth one deeper than the input b_i
    Sample Call:
        final_array = branch_recurse(x_array,dataset_bins,b_i,depth,curr_depth)

    """
    if depth - 2 > curr_depth:
        final_array = []
        curr_len = np.shape(b_i)[0]
        # Loop through all the subclasses for our current depth
        #	Call the recursive function again: go a branch deeper into the tree 
        #to get all the subclasses for a different predictive variable
        #	Note: x_array and dataset_bins don't change in the new call - these 
        #are relevant for the branches we wish to ADD to the tree
        #	We do give only a subset of b_i - we pass only the relevant array 
        #indices for each subclass i of the current variable.
        for i in range(curr_len): 
            final_array.append(branch_recurse(x_array,dataset_bins,b_i[i],depth,curr_depth + 1))
    else: # We now arrive at the end of the current tree - create branches!

         prov_array = branch_fill_func(x_array,dataset_bins)
         final_array = [None]*len(b_i) #The remaining code in the base case finds the indices of each subclass of these branches (that's in prov array), and intersects them with the relevant class of x in the layer above (that's in b_i) - hence the structure of b_i
         for i in range(len(b_i)):
             final_array[i] = [None]*len(prov_array)

         for root_i in range(len(b_i)):
             for prov_i in range(len(prov_array)):
                 final_array[root_i][prov_i] = np.intersect1d(prov_array[prov_i],b_i[root_i])
         print 'Done with recursion'
    return final_array

def entropy_recurse(y_vals,x_vals,dataset_bins,b_i,depth,curr_depth):
    """
    Recursively create branches for a tree
    
    Inputs
    ------
    y_vals : array
        The output values, assumed to be either 1 or 0
    x_array : array
        The independent variable for which we want to calculate the entropy
    dataset_bins : list
        The classes of x_array
    b_i : list
        A list with the same shape as the tree that contains the indices for each class
        First dimension - the indices for each of the different classes of tree root x_1 = i b_i[0] = all indices where x_1 = 0...
        Second dimension - subclass: b_i[0,0] = all indices where x_1 = 0, x_2 = 0, b_i[3,4] = all indices where x_1 = 3, x_2 = 4
        Third dimension - etc... (b_i[2,5,3] = all indices where x_1 = 2, x_2 = 5, x_3 = 3...
    depth : integer
        Current depth of full tree
    curr_depth : integer
        Depth of entropy calculation depth of tree

    Returns
    -------
    ent : integer
        The entropy of the data for the current variable x_array
    class_size : integer
        The class size of the subclass in the base case.
            This is not needed in the original call of entropy_recurse, but IS needed to get the entropy during the recursive calls!
    
    Sample Call:
    ------------
    [ent,class_size] = entropy_recurse(y_vals,x_array,dataset_bins,b_i,depth,curr_depth)

    """
    ent = 0.0
    class_size = 0
    if depth > curr_depth:
        curr_len = np.shape(b_i)[0]
        # Loop over all classes of the x in this branch of the current layer
        for i in range(curr_len): 
            ent_c,c_c = entropy_recurse(y_vals,x_vals,dataset_bins,b_i[i],
                                        depth,curr_depth+1) 
            ent = ent + ent_c
            #Sum up the size of the class, needed to calculate the probability we are in class i of variable x
            class_size = class_size + c_c  
        if class_size >0: #Else case unnecessary, as that shouldn't contribute to entropy at all.
            ent = ent/class_size
    else: # Base case, we are at the end of the tree
        class_size = len(b_i)
        y_class = np.array([y_vals[kk] for kk in b_i]) # Relevant indep variable for subclass
        x_class = np.array([x_vals[kk] for kk in b_i]) # Relevant dep variable for subclass
        if len(y_class) > 0: # Go back and get the entropty
            ent = class_size*entropy_calc(y_class,x_class,dataset_bins)
        else: #This class is empty, no entropy contribution
            ent = 0.0
    return ent,class_size

def branch_fill_func(data,vals):
    """
    Recursively create branches for a tree
    
    Inputs
    ------
    data : array
        The independent variable x
    vals : array
        The classes of data

    Returns
    -------
    branch_fill : list
        A list with length vals that gives the global indices relevant for each subclass of data
    
    Sample Call
    -----------
    branch_fill = branch_fill_func(data,vals)

    """
    b = len(vals)
    branch_fill = [None]*b
    for x in range(b): # Find indices for each subclass of the data set.
        branch_fill[x] = [kk for kk in np.where(data == vals[x])][0]
    return branch_fill
