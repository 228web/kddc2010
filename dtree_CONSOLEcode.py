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


#pp = [sd_d,o_ds]
#pp_bins = [[i for i in range(1,len(stepdur_bins)+1)],[i for i in range(1,len(ob_sparse)+1)]]

dataset = [o_ds,sd_d,np.array(xy_train['Hints'])]
dataset_bins = [[i for i in range(1,len(ob_sparse)+1)],[i for i in range(1,len(stepdur_bins)+1)],[i for i in range(0,max(xy_train['Hints'])+1)]]
var_strings = ['Opportunity','Step Duration','Hints']
#tree_order = [0,1,2]

#final_branch = calc_dtree(y_pred,dataset,dataset_bins,var_strings,tree_order)
[final_branch,tree_order] = calc_adapt_dtree(y_pred,dataset,dataset_bins,var_strings)

tree_shape = np.shape(final_branch)
pred_array = np.zeros((tree_shape),int)

#Hard coded 2 level tree
for i in range(tree_shape[0]):
    for j in range(tree_shape[1]):
        y_curr = np.array([y_pred[kk] for kk in final_branch[i][j]])
        class_y = sum(y_curr)/float(len(y_curr))
        pred_array[i][j] = np.round(class_y)


#Hard coded 3 level tree
for i in range(tree_shape[0]):
    for j in range(tree_shape[1]):
        for k in range(tree_shape[2]):
            y_curr = np.array([y_pred[kk] for kk in final_branch[i][j][k]])
            if len(y_curr) > 0:
                class_y = sum(y_curr)/float(len(y_curr))
                pred_array[i][j][k] = np.round(class_y)
            else: # This is class is empty, so it doesn't matter what we do.
                pred_array[i][j][k] = 0


y_out = np.zeros((nsize),int)

for k in range(nsize):
    bin0 = dataset_bins[tree_order[0]].index(dataset[tree_order[0]][k])
    bin1 = dataset_bins[tree_order[1]].index(dataset[tree_order[1]][k])
    bin2 = dataset_bins[tree_order[2]].index(dataset[tree_order[2]][k])

    y_out[k] = pred_array[bin0][bin1-1][bin2-1]

np.square((y_pred - y_out)).sum()/float(nsize)
