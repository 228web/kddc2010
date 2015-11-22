y_pred = xy_train['Correct First Attempt']

ent_base = dt.entropy_calc(y_pred,[0],[])

opp_sqz = np.zeros((nsize),int)
for i in range(nsize):
#    hh = [x for x in opp_digit[i,:] if x > 1]
    if sum(opp_array[i,:]) > 0.0:
        hh = [x for x in opp_array[i,:] if x > 0]
        opp_sqz[i] = np.min(hh)


opp_bins = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
ob_sparse = [0,1,5,10,15]
#    ob_sparse = [0,1,15]

o_d = np.digitize(opp_sqz,opp_bins)
o_ds = np.digitize(opp_sqz,ob_sparse)

stepdur_bins = [0,1,15,30,45,60]
sd_d = np.digitize(xy_train['Step Duration (sec)'],stepdur_bins)


p_strings = ['Step Duration','Opportunity']


pp = [sd_d,o_ds]
pp_bins = [[i for i in range(1,len(stepdur_bins)+1)],[i for i in range(1,len(ob_sparse)+1)]]

dataset = [o_ds,sd_d]#,xy_train['Hints']]
dataset_bins = [[i for i in range(1,len(ob_sparse)+1)],[i for i in range(1,len(stepdur_bins)+1)]]#,[i for i in range(0,max(xy_train['Hints'])+1)]]
var_strings = ['Opportunity','Step Duration']#,'Hints']
tree_order = [1,0]#,2]

final_branch = calc_dtree(y_pred,dataset,dataset_bins,var_strings,tree_order)

tree_shape = np.shape(final_branch)
pred_array = np.zeros((tree_shape),int)

#Hard coded 2 level tree for now
for i in range(tree_shape[0]):
    for j in range(tree_shape[1]):
        y_curr = np.array([y_pred[kk] for kk in final_branch[i][j]])
        class_y = sum(y_curr)/float(len(y_curr))
        pred_array[i][j] = np.round(class_y)


y_out = np.zeros((nsize),int)

for k in range(nsize):
    bin0 = dataset_bins[tree_order[0]].index(dataset[tree_order[0]][k])
    bin1 = dataset_bins[tree_order[1]].index(dataset[tree_order[1]][k])

    y_out[k] = pred_array[bin0-1][bin1-1]

np.square((y_pred - y_out)).sum()/float(nsize)





def calc_dtree(y_vals,dataset,dataset_bins,var_strings,tree_order):
    """
    Given a data set, and the strings of a dictionary, return a decision tree with the order given in the string
    
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
#            old_shape = shape_array
#            old_branch_indices = branch_indices[0:d+1]
#            shape_array = np.append(old_shape_array,num_branches)
            shape_array = np.append(shape_array,num_branches)
            branch_indices = branch_recurse(dataset[b_index],dataset_bins[b_index],branch_indices,d+1,0)

    return branch_indices



def branch_recurse(x_array,dataset_bins,b_i,depth,curr_depth):
    if depth - 2 > curr_depth: # This is not 100% working yet, but should not take too much tweaking.
        final_array = []
        curr_len = len(dataset_bins[curr_depth])
        print 'Current depth is ' + str(curr_depth)
        print 'Expected loop is ' + str(curr_len)
        for i in range(curr_len):
            print i
            print len(b_i[i])
#            final_array = [final_array,branch_recurse(x_array,dataset_bins,branch_indices[i],depth,curr_depth + 1)]
            final_array[i] = branch_recurse(x_array,dataset_bins,branch_indices[i],depth,curr_depth + 1)

    else:
         prov_array = branch_fill_func(x_array,dataset_bins)
         final_array = [None]*len(b_i)
         for i in range(len(b_i)):
             final_array[i] = [None]*len(prov_array)
         for root_i in range(len(b_i)):
#             print b_i[root_i]
             for prov_i in range(len(prov_array)):
#                 print(prov_array[prov_i])
                 final_array[root_i][prov_i] = np.intersect1d(prov_array[prov_i],b_i[root_i])
         print 'Done with recursion'
    return final_array

def branch_fill_func(data,vals):
    b = len(vals)
    branch_fill = [None]*b
    for x in range(b):
        branch_fill[x] = [kk for kk in np.where(data == vals[x])][0]
    return branch_fill







#This is an old version of calc_dtree. I am leaving it here for now as long as I think I might need to scavenge code, but otherwise ignore it.

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
    ob_sparse = [0,1,5,10,15]
#    ob_sparse = [0,1,15]


    o_d = np.digitize(opp_sqz,opp_bins)
    o_ds = np.digitize(opp_sqz,ob_sparse)


    stepdur_bins = [0,1,15,30,45,60]
    sd_d = np.digitize(xy_train['Step Duration (sec)'],stepdur_bins)


    p_strings = ['Step Duration','Opportunity']



    pp = [sd_d,o_ds]
    pp_bins = [[i for i in range(1,len(stepdur_bins)+1)],[i for i in range(1,len(ob_sparse)+1)]]

    pp = [sd_d,o_d]
    pp_bins = [[i for i in range(1,len(stepdur_bins)+1)],[i for i in range(1,len(opp_bins)+1)]]




    branches = []


    td = 2

    y_curr = y_pred
    for dep in range(td):
        IG = ent_base
        for i in range(len(pp_remain)):
            i_gain_prov = dt.entropy_calc(y_curr,pp[i],pp_bins[i])
            IG = min(IG,ent_base - i_gain_prov)
        branch_point = IG.index(np.max(IG))
#        branches.append(p_strings[branch_point])
#        b_inds.append(branch_point)
#        del b_remain[branch_point]
#        ent_base = min(eb)

    b_inds = [0,1]

    pred_array = np.zeros((len(pp_bins[b_inds[0]]),len(pp_bins[b_inds[1]])),int)
    ent_c = 0.0

    lev_classes = pp_bins[b_inds[0]]
    for lc in range(1,len(lev_classes)+1):
        loo = [kk for kk in np.where(pp[b_inds[0]] == lc)][0]
        y_lev = np.array([y_pred[kk] for kk in loo])
        x_lev = np.array([pp[b_inds[0]][kk] for kk in loo])

        csize = len(y_lev)
        print 'Big class size is '+ str(csize)

        print 'Entropy is ' + str(dt.entropy_calc(y_lev,x_lev,pp_bins[b_inds[1]]))
        ent_c = ent_c + (csize/float(nsize))*dt.entropy_calc(y_lev,x_lev,pp_bins[b_inds[1]])
        d2class = pp_bins[b_inds[1]]
        for lccc in range(1,len(d2class)+1):
            c2_array = [kk for kk in np.where(pp[b_inds[1]][loo] == lccc)][0]
#            print c2_array
            y_2class = np.array([y_lev[kk] for kk in c2_array])
            x_2class = np.array([x_lev[kk] for kk in c2_array])
            c2size = len(y_2class)
            class_y = sum(y_2class)/float(sum(y_lev))
            print class_y
#            print c2size
            pred_array[lc-1,lccc-1] = np.round(class_y)
#            print 'Small class size is '+ str(c2size)

    y_out = np.zeros((nsize),int)

    for k in range(nsize):
        bin1 = pp[b_inds[1]][k]
        bin0 = pp[b_inds[0]][k]

#        print 'Bin0 is ' + str(bin0)
#        print 'Bin1 is ' + str(bin1)
#        if bin0 - 1 < 0:
#            y_out[k] = 0.0
#        elif bin1 - 1 < 0:
#            y_out[k] = 0.0
#        else:
        y_out[k] = pred_array[bin0-1][bin1-1]
#        print y_out[k]

    np.square((y_pred - y_out)).sum()/float(nsize)

# Fix this later
#        print y_lev

#        print dt.entropy_calc(np.array(y_lev),np.array(x_lev),pp_bins[b_remain[0]])
#        print csize/float(nsize)*dt.entropy_calc(np.array(y_lev),np.array(x_lev),pp_bins[b_remain[0]])

#    ent_lev = ent_lev + csize/float(nsize)*dt.entropy_calc(np.array(y_lev),np.array(x_lev),pp_bins[b_remain[0]])


#        else:
#            lev_classes = pp_bins[b_inds[0]]
#            for lc in range(len(lev_classes)):
#                loo = [kk for kk in np.where(pp[b_inds[0]] == lc)][0]
#                y_lev = [y_pred[kk] for kk in loo]
#                x_lev = [pp[b_remain[0]][kk] for kk in loo]
