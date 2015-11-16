#!/usr/bin/python

import loader as ld
import tagger as tg
import ID_assigner as ida
import numpy as np
import dtree as dt

[xy_keys,xy_train] = ld.loader(ld.trainDat)

time_strings = ['Step Start Time','First Transaction Time','Correct Transaction Time','Step End Time']

for i in range(4):
    print 'Processing ' + time_strings[i]
    xy_train[time_strings[i]] = ld.convert_times(xy_train[time_strings[i]])

id_strings = ['Anon Student Id','Problem Name']
all_dicts = []

for i in range(2):
    print 'Processing ' + id_strings[i]
    [xy_train[id_strings[i]],temp] = ida.ID_assigner(xy_train[id_strings[i]])
    all_dicts.append(temp)

[xy_train['Problem Hierarchy'],temp,temp2] = ida.unit_ID_assigner(xy_train['Problem Hierarchy'])
all_dicts.append(temp)
all_dicts.append(temp2)

#All_dicts:
#	0: Student ID
#	1: Problem Name
#	2: Problem Unit
#	3: Problem Section

tag_master = tg.string_tags(xy_train['KC(Default)'])

[tag_array,opp_array] = tg.tags_to_array(xy_train['KC(Default)'],xy_train['Opportunity(Default)'],tag_master)

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

#Look up location of index in array
#tag_master.index(SOME STRING)

y_pred = xy_train['Correct First Attempt']

#Check entropy of the data
ent = dt.entropy_calc(map(int,y_pred),[0],[])

#ent = entropy_calc(map(int,y_pred),x_prob,all_dicts[1].keys())

# Calculate step start time from FIRST student transaction
# Calculate the above, but separate by skill?

