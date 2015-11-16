#!/usr/bin/python

import numpy as np

#Sanity check - do multiple skills correspond to multiple non-zero opportunities?
def data_test():
    for i in range(len(tag_array)):
        if(np.count_nonzero(opp_array[i,:]) != np.sum(tag_array[i,:])):
            print 'ERROR: Opp does not contain correct number of entries: ' + str(i)

#Another sanity check - does opportunity increment for each instance of a skill FOR each student
def skill_test():
    for i in range(len(tag_master)):
        skill_locs = np.where(tag_array[:,i] > 0)
        print [opp_array[j,i] for j in skill_locs]
        a =  [opp_array[j,i] for j in skill_locs]
