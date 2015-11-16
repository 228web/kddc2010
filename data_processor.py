#!/usr/bin/python

import loader as ld
import tagger as tg
import ID_assigner as ida
import numpy as np
import dtree as dt

[xy_keys,xy_train] = ld.loader(ld.trainDat)

nsize = len(xy_train['Anon Student Id'])

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
int_s = ['Correct First Attempt','Incorrects','Hints','Corrects']

for i in range(len(int_s)):
    xy_train[int_s[i]] = map(int,xy_train[int_s[i]])

y_pred = xy_train['Correct First Attempt']

#Check entropy of the data
ent = dt.entropy_calc(y_pred,[0],[])

def step_normalize(stud_IDs,stud_dict,step_start_time,first_trans_time,corr_trans_time,step_end_time):
# Normalizes the step start time by student's first transation time
    aa = np.copy(step_start_time)
    bb = np.copy(first_trans_time)
    dd = np.copy(step_end_time)
    cc = np.copy(corr_trans_time)
    for stud in stud_dict:
        print('Processing student ' + str(stud))
        rel_steps = [step_start_time[i] for i in np.where(stud_IDs == stud)][0]
        rel_ind = np.where(stud_IDs == stud)[0]

# In case this array isn't sorted...
        rel_steps_ind_sort = np.argsort(rel_steps)
        fnz = [i for i in rel_steps if i > 0]
        zmin =  np.where(rel_steps == min(fnz))[0]

        ftime = rel_steps[zmin] - 1


        sst = step_start_time[rel_ind]
        ftt = first_trans_time[rel_ind]
        ctt = corr_trans_time[rel_ind]
        se = step_end_time[rel_ind]

#        print np.where(ctt > 0.0, 9, 11)
        print ctt - ftime

        aa[rel_ind] = np.where(sst > 0.0,sst - ftime,0.0)
        bb[rel_ind] = np.where(ftt > 0.0,ftt - ftime,0.0)
        cc[rel_ind] = np.where(ctt > 0.0,ctt - ftime,0.0)
        dd[rel_ind] = np.where(se  > 0.0,se  - ftime,0.0)

        print cc[rel_ind[:10]]

    return aa,bb,cc,dd

[aa,bb,cc,dd] = step_normalize(xy_train['Anon Student Id'],all_dicts[0].keys()[1:],xy_train['Step Start Time'],xy_train['First Transaction Time'],xy_train['Correct Transaction Time'],xy_train['Step End Time'])

#I think this is working???
#[xy_train['Step Start Time'],xy_train['First Transaction Time'],xy_train['Correct Transaction Time'],xy_train['Step End Time']] = step_normalize(xy_train['Anon Student Id'],all_dicts[0].keys()[1:],xy_train['Step Start Time'],xy_train['First Transaction Time'],xy_train['Correct Transaction Time'],xy_train['Step End Time'])


#for i in range(len(step_norm_list)):
#    xy_train[step_norm_list[i]] = step_normalize(xy_train['Anon Student Id'],all_dicts[0].keys()#[1:],xy_train[step_norm_list[i]])

#ent = entropy_calc(map(int,y_pred),x_prob,all_dicts[1].keys())

