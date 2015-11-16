# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:25:43 2015

@author: John
"""
import hmm as hmm
import numpy as np
import loader as ld
import tagger as tg
import ID_assigner as ida
import matplotlib.pyplot as plt

time_strings = ['Step Start Time','First Transaction Time',
                'Correct Transaction Time','Step End Time']
id_strings = ['Anon Student Id','Problem Name']

def processor(data):
    [xy_keys,xy_train] = ld.loader(data)
    
    # Process time strings to seconds
    #for i in range(4):
    #    print 'Processing ' + time_strings[i]
    #    xy_train[time_strings[i]] = ld.convert_times(xy_train[time_strings[i]])

    # Dictionary of anonId and problem tags
    all_dicts = []

    for i in range(2):
        print 'Processing ' + id_strings[i]
        [xy_train[id_strings[i]],temp] = ida.ID_assigner(xy_train[id_strings[i]])
        all_dicts.append(temp)

    [xy_train['Problem Hierarchy'],temp,temp2] = ida.unit_ID_assigner(
                                                xy_train['Problem Hierarchy'])
    all_dicts.append(temp)
    all_dicts.append(temp2)
    
    tag_master = tg.string_tags(xy_train['KC(Default)'])

    [tag_array,opp_array] = tg.tags_to_array(
                                            xy_train['KC(Default)'],
                                            xy_train['Opportunity(Default)'],
                                            tag_master)
    
    
    return xy_keys, xy_train, tag_master, tag_array, opp_array
    
xy_keys, xy_train, tags, tagA, oppA = processor(ld.trainDat)
tagLen = len(tags)
datLen = len(xy_train[xy_keys[1]])

idSplit = [k+1 for k in range(len(xy_train[xy_keys[0]])-1) if
            xy_train[xy_keys[1]][k+1]!= xy_train[xy_keys[1]][k]]
# data as array, [studentID, firstCorrect, hints, incorrects, corrects]
data = np.zeros([datLen, 5])
data[:,0] = xy_train[xy_keys[1]]
data[:,1] = np.array(xy_train['Correct First Attempt'],dtype=int)
#data[:,2] = np.array(xy_train['Hints'],dtype=int)
data[:,3] = np.array(xy_train['Incorrects'],dtype=int)
data[:,4] = np.array(xy_train['Corrects'],dtype=int)

observations = np.zeros(datLen)
"""
Start assuming we have only two states Know and Dunno
Start probabilities are 90% start as Dunno
Fake transition probabilities:[[.8,.2],[.01,.99]]
IE assume they can move from Dunno to Know with 20% prob and stay Dunno with 
80% prob. And once they Know, they stay at Know with 99% and only go back with 
1%.
What will they emit:
first correct, hints, incorrects (hints and incorrects need to get binned)
 (hints!=0 and incorrects <1)
say if they Know then first corrects will be [.1, .9] (get it right first try 90%)
and hints at [.2, .8]
and incorrects as [.1, .9]

So 8 emission states: first correct = [0,1], hints = [yes, no], and incorrects = [yes, no]

Similarly if dunno:
first corrects: [.5, .5]
hints: [.7, .3]
incorrects: [.5, .5]
"""
def normalize(dataArray):
    length = len(dataArray)
    norm = np.zeros(length)
    for k in range(dataArray):
        if dataArray[k] != 0:
            norm[k] = 1
    return norm
    
def smarter(correctData, incorrectData):
    length = len(correctData)
    smart = np.zeros(length)
    for k in range(length):
        if correctData[k] > .9*incorrectData[k]:
            smart[k] = 1
    return smart
    
data[:,4] = smarter(data[:,4],data[:,3])

for k in range(datLen):
    if data[k,4]>0:
        observations[k] += 1
    if data[k,1]>0:
        observations[k] += 1

numStud = len(idSplit)+1


def hmm_tester(x, start, trans, emit):
    predicts = np.zeros(numStud)
    rmse = np.zeros(numStud)
    f,b,probF,probB,post = hmm.frwd_bkwd(observations[:idSplit[0]-1],startP,transP,emitP)
    predicts[0] = np.dot(emitP[:,2],np.dot(transP,post[-1]))
    rmse[0] = np.sqrt((data[idSplit[0]-1,1]-predicts[0])**2)
    f,b,probF,probB,post = hmm.frwd_bkwd(observations[idSplit[-1]:],startP,transP,emitP)
    predicts[-1] = np.dot(emitP[:,2],np.dot(transP,post[-1]))
    rmse[-1] = np.sqrt((data[-1,1]-predicts[-1])**2)
    for k in range(numStud-2):
        f,b,probF,probB,post = hmm.frwd_bkwd(observations[idSplit[k]:idSplit[k+1]],startP,transP,emitP)
        predicts[k] = np.dot(emitP[:,2],np.dot(transP,post[-1]))
        rmse[k] = np.sqrt((data[idSplit[k]-1,1]-predicts[k])**2)
        
    return rmse

error = np.zeros(20)
for k in range(20):
    p = k*.01+.79
    startP = [p, 1-p]
    transP = [[.8,.2],[.01,.99]]
    emitP = np.array([[.6,.3,.1],[.1,.3,.6]])
    rmse = hmm_tester(observations, startP, transP, emitP)
    predictions = [rmse[l] for l in range(numStud) if not np.isnan(rmse[l])]
    error[k] = 1-np.average(predictions)
    
plt.scatter(np.arange(20),error)
    