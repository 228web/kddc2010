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

for i in range(2):
    print 'Processing ' + id_strings[i]
    xy_train[id_strings[i]] = ID_assigner(xy_train[id_strings[i]])

xy_train['Problem Hierarchy'] = unit_ID_assigner(xy_train['Problem Hierarchy'])

tag_master = string_tags(xy_train['KC(Default)'])

[tag_array,opp_array] = tags_to_array(xy_train['KC(Default)'],xy_train['Opportunity(Default)'],tag_master)

#Sanity check - do multiple skills correspond to multiple arrays?
mult_skill_locs = np.where(np.sum(tag_array,1)>1)


# Calculate step start time from FIRST student transaction
# Calculate the above, but separate by skill?

