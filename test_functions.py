#!/usr/bin/python

import numpy as np
import dtree as dt

#Note, arguments need to be added to these tests...

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

#Some simple decision tree testing
def decision_tree_tests():
    xs1 = np.array([1,1,1,1,1,1])
    xs2 = np.array([1,1,0,0,1,0])
    xs3 = np.array([1,0,1,0,1,0])
    xs4 = np.array([1,0,0,0,1,0])
    ys  = np.array([1,1,0,0,1,0])

    ent_base = dt.entropy_calc(ys,[],[])
    ent_x1 = dt.entropy_calc(ys,xs1,[0,1])
    ent_x2 = dt.entropy_calc(ys,xs2,[0,1])
    ent_x3 = dt.entropy_calc(ys,xs3,[0,1])
    ent_x4 = dt.entropy_calc(ys,xs4,[0,1])

    ent_x3_2 = dt.entropy_calc(ys,xs3,[0,1,2])

    if ent_base != 1.0:
        print 'ERROR: Base case for entropy calculation incorrect.'
    if ent_x1 != 1.0:
        print 'Error: Random data should give entropy = 1!'
        print 'Calculated entropy is ' + str(ent_x1)
    if ent_x2 != 0.0:
        print 'Error: Perfect data should give entropy = 0!'
        print 'Calculated entropy is ' + str(ent_x2)
    if ent_x3 < ent_x2:
        print 'Error: Imperfect data should do worse than perfect data!'
        print 'Malformed expression: ' + str(ent_x3) + '<' + str(ent_x2)
    if ent_x3 > ent_x1:
        print 'Error: Imperfect data should do better than random data!'
        print 'Malformed expression: ' + str(ent_x3) + '>' + str(ent_x1)
    if ent_x3 < ent_x4:
        print 'Error: Malformed expression: ' + str(ent_x3) + '<' + str(ent_x4)
    if ent_x3_2 != ent_x3:
        print 'Error: Empty class should not matter'

