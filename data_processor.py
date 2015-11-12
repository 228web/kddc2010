#!/usr/bin/python

import numpy as np
import datetime as dt
from collections import defaultdict

[xy_keys,xy_train] = loader(trainDat)

time_strings = ['Step Start Time','First Transaction Time','Correct Transaction Time','Step End Time']

for i in range(4):
    print 'Processing ' + time_strings[i]
    xy_train[time_strings[i]] = convert_times(xy_train[time_strings[i]])

id_strings = ['Anon Student Id','Problem Name']
all_dicts = []

for i in range(2):
    print 'Processing ' + id_strings[i]
    [xy_train[id_strings[i]],temp] = ID_assigner(xy_train[id_strings[i]])
    all_dicts.append(temp)

[xy_train['Problem Hierarchy'],temp,temp2] = unit_ID_assigner(xy_train['Problem Hierarchy'])
all_dicts.append(temp)
all_dicts.append(temp2)

#All_dicts:
#	0: Student ID
#	1: Problem Name
#	2: Problem Unit
#	3: Problem Section

tag_master = string_tags(xy_train['KC(Default)'])

[tag_array,opp_array] = tags_to_array(xy_train['KC(Default)'],xy_train['Opportunity(Default)'],tag_master)

#Sanity check - do multiple skills correspond to multiple non-zero opportunities?


for i in range(len(tag_array)):
    if(np.count_nonzero(opp_array[i,:]) != np.sum(tag_array[i,:])):
        print 'ERROR: Opp does not contain correct number of entries: ' + str(i)

#Another sanity check - does opportunity increment for each instance of a skill FOR each student
for i in range(len(tag_master)):
    skill_locs = np.where(tag_array[:,i] > 0)
    print [opp_array[j,i] for j in skill_locs]
    a =  [opp_array[j,i] for j in skill_locs]

#Look up location of index in array
#tag_master.index(SOME STRING)

#Write a decision tree

# Calculate step start time from FIRST student transaction
# Calculate the above, but separate by skill?

