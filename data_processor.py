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

#id_strings = ['Anon Student Id','Step Name','Problem Name']

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

def step_start_normalize(stud_IDs,step_start_times):
# Normalizes the step start time by student's first transation time
    sst_T = np.copy(step_start_times)
    for stud in stud_IDs:
        rel_steps = [step_start_times[i] for i in np.where(stud_IDs == stud)]
#        print(rel_steps[0])

# In case this array isn't sorted...
        rel_steps_ind_sort = (np.argsort(rel_steps)[0])

        for i in range(len(rel_steps_ind_sort)):
            step_start_times[rel_steps_ind_sort[i]] = step_start_times[rel_steps_ind_sort[i]] - step_start_times[rel_steps_ind_sort[0]]

    return step_start_times


#ent = entropy_calc(map(int,y_pred),x_prob,all_dicts[1].keys())
