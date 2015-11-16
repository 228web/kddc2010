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

id_strings = ['Anon Student Id','Step Name','Problem Name']
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
#	1: Step Name
#	2: Problem Name
#	3: Problem Unit
#	4: Problem Section

tag_master = tg.string_tags(xy_train['KC(Default)'])

[tag_array,opp_array] = tg.tags_to_array(xy_train['KC(Default)'],xy_train['Opportunity(Default)'],tag_master)

#Look up location of index in array
#tag_master.index(SOME STRING)

y_pred = xy_train['Correct First Attempt']

#Check entropy of the data
ent = dt.entropy_calc(map(int,y_pred),[0],[])

#ent = entropy_calc(map(int,y_pred),x_prob,all_dicts[1].keys())

#Some variables we could split on:
#	Anon Student ID
#	Problem Unit
#	Problem Section
#	Problem View
#	Problem Name
#	Step Name
#	Step Start Time (normalized by FIRST student transaction time)
#	Step Start Time (normalized by FIRST student transaction time for each skill)
#	Step Duration
#	Incorrects
#	Hints
#	Skill
#	Opportunity

# Calculate step start time from FIRST student transaction
# Calculate the above, but separate by skill?

