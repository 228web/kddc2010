# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:25:43 2015

@author: John
"""

"""
This is a script to test the file hmm.py.

It currently uses the 2005-06 data to train whether each student is in a 
knowledgeable or "dunno" state as a hidden state.

Description of HMM states
Start assuming we have only two hidden states: [Dunno, Know]
startP = [.9,.1] respectively
Because we don't know tranisition probabilities, we fake them.
Fake transition probabilities:[[.8,.2],[.01,.99]]
IE assume they can move from Dunno to Know with 20% prob and stay Dunno with 
80% prob. And once they Know, they stay at Know with 99% and only go back with 
1%.

Again, we fake emission probabilites:
Observable states are: [first correct, corrects > incorrects, otherwise]

say if they Know then first corrects will be [.1, .9] (get it right first try 90%)
and c>i at [.2, .8]
and incorrects as [.1, .9]

Similarly if dunno:
first corrects: [.5, .5]
c>i: [.7, .3]
incorrects: [.5, .5]
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
    """
    This is a functional form of data_processor sans some features.
    
    Inputs
    ------
    data : string
        path and file name of data (e.g. ld.trainDat)
        
    Returns
    -------
    xy_keys : list
        list of keys for data dictionary
    dat_array : ndarray
        parsed data dictionary now stored in numpy array
    tag_master : list
        list of tags from Knowledge Component data
    tag_array : ndarray
        array of knowledge component presence in each question
    opp_array : ndarray
        array of opportunity count for each component in each question
    """
    xy_keys,xy_train = ld.loader(data)
    
    # Process time strings to seconds
    for i in range(4):
        print 'Processing ' + time_strings[i]
        xy_train[time_strings[i]] = ld.convert_times(xy_train[time_strings[i]])
        
    # Convert Step Duration to seconds
    xy_train['Step Duration (sec)'] = (xy_train['Step End Time']-
                                        xy_train['Step Start Time'])

    # Dictionary of anonId and problem tags
    all_dicts = []

    # Process string ids
    for i in range(2):
        print 'Processing ' + id_strings[i]
        xy_train[id_strings[i]],temp = ida.ID_assigner(xy_train[id_strings[i]])
        all_dicts.append(temp)

    xy_train['Problem Hierarchy'],temp,temp2 = ida.unit_ID_assigner(
                                                xy_train['Problem Hierarchy'])
    all_dicts.append(temp)
    all_dicts.append(temp2)
    
    #These are the variables I care about at the moment, add if want more - JGL
    # 'Anon Student Id','Incorrects','Corrects','Problem View',
    #'Correct Transaction Time','Correct First Attempt','Step Start Time',
    #'First Transaction Time','Problem Hierarchy','Hints','Step End Time']
    # KC(Default) and Opportunity(Default) separate arrays.
    dat_array = np.empty([datLen,14])
    dat_array[:,0] = xy_train['Anon Student Id']
    dat_array[:,1] = xy_train['Problem Name']
    dat_array[:,2] = xy_train['Problem Hierarchy']
    dat_array[:,3] = np.array(xy_train['Incorrects'],dtype=int)
    dat_array[:,4] = np.array(xy_train['Hints'],dtype=int)
    dat_array[:,5] = np.array(xy_train['Corrects'],dtype=int)
    dat_array[:,6] = np.array(xy_train['Correct First Attempt'],dtype=int)
    dat_array[:,7] = np.array(xy_train['Problem View'],dtype=int)
    dat_array[:,8] = xy_train['Step Start Time']
    dat_array[:,9] = xy_train['First Transaction Time']
    dat_array[:,10] = xy_train['Correct Transaction Time']
    dat_array[:,11] = xy_train['Step End Time']
    dat_array[:,12] = xy_train['Step Duration (sec)']
    dat_array[:,13] = ld.check_final_answer(xy_train['Step Name'])
    
    # Process Knowledge components
    tag_master = tg.string_tags(xy_train['KC(Default)'])

    # Process opportunity
    tag_array,opp_array = tg.tags_to_array(
                                xy_train['KC(Default)'],
                                xy_train['Opportunity(Default)'],
                                tag_master)
    
    
    return xy_keys, dat_array, tag_master, tag_array, opp_array
    
def normalize(dataArray):
    length = len(dataArray)
    norm = np.zeros(length)
    for k in range(dataArray):
        if dataArray[k] != 0:
            norm[k] = 1
    return norm
    
def smarter(correctData, incorrectData):
    """
    Function to process whether a student received more corrects than 
    incorrects on a particular question.
    
    Inputs
    ------
    correctData : ndarray
        correctData column
    incorrectData : ndarray
        incorrect Data column
        
    Returns
    -------
    smart : ndarray
        1 or 0 depending on whether more corrects than .9*incorrects each step
    """
    length = len(correctData)
    smart = np.zeros(length)
    for k in range(length):
        if correctData[k] > .9*incorrectData[k]:
            smart[k] = 1
    return smart
    
#The following is just a script from before the milestone to test the forward-
#backward algorithm and its predictions

#Import data
xy_keys, xy_train, tags, tagA, oppA = processor(ld.trainDat)

#relevant scales
tagLen = len(tags)
datLen = len(xy_train[xy_keys[1]])

#Locations of splits by student id
idSplit = [k+1 for k in range(len(xy_train[xy_keys[0]])-1) if
            xy_train[xy_keys[1]][k+1]!= xy_train[xy_keys[1]][k]]

#Relevant data for hmm test based on earlier definitions
data = np.zeros([datLen, 5])
data[:,0] = xy_train[xy_keys[1]]
data[:,1] = np.array(xy_train['Correct First Attempt'],dtype=int)
data[:,2] = np.array(xy_train['Hints'],dtype=int)
data[:,3] = np.array(xy_train['Incorrects'],dtype=int)
data[:,4] = np.array(xy_train['Corrects'],dtype=int)

observations = np.zeros(datLen)

#Reprocess corrects into more corrects than incorrects data
data[:,4] = smarter(data[:,4],data[:,3])

#Create observation data states
#[0,1,2] = [otherwise, c>i, correct]
for k in range(datLen):
    if data[k,4]>0:
        observations[k] += 1
    if data[k,1]>0:
        observations[k] += 1

#Number of students
numStud = len(idSplit)+1


def hmm_tester(x, start, trans, emit):
    """
    This is a function to test the forward-backward algorithm in hmm.py. Splits 
    by student and runs f-b on all steps up to n-1, then compares prediction to
     nth step
    
    Inputs
    ------
    x : ndarray
        training observation data, nx1
    start : ndarray
        starting probabilities, kx1
    trans : ndarray
        transition probabilities, kxk
    emit : ndarray
        emission probabilities, kxd
        
    Returns
    -------
    rmse : ndarray
        array of root-mean-square-error on prediction of first correct on next 
        question compared to actual next data point result. This is currently 
        not using the test data.
    """
    #Initialize array of predictions, probability of correct on next question
    predicts = np.zeros(numStud)
    
    #Initialize array for rmse to compare to actual test data
    rmse = np.zeros(numStud)
    
    #Run forward-backward on first student
    f,b,probF,probB,post = hmm.frwd_bkwd(observations[:idSplit[0]-1],
                                         startP,transP,emitP)
    #Predict and compute error on first student
    predicts[0] = np.dot(emitP[:,2],np.dot(transP,post[-1]))
    rmse[0] = np.sqrt((data[idSplit[0]-1,1]-predicts[0])**2)
    
    #Run forward-backward on last student
    f,b,probF,probB,post = hmm.frwd_bkwd(observations[idSplit[-1]:-1],
                                         startP,transP,emitP)
                                         
    #Predict and compute error on last student
    predicts[-1] = np.dot(emitP[:,2],np.dot(transP,post[-1]))
    rmse[-1] = np.sqrt((data[-1,1]-predicts[-1])**2)
    
    #Run fwd-bkwd, predict, and compute error on remaining students
    for k in range(numStud-2):
        f,b,probF,probB,post = hmm.frwd_bkwd(
                                    observations[idSplit[k]:idSplit[k+1]-1],
                                    startP,transP,emitP)
        predicts[k] = np.dot(emitP[:,2],np.dot(transP,post[-1]))
        rmse[k] = np.sqrt((data[idSplit[k]-1,1]-predicts[k])**2)
        
    return rmse
"""
error = np.zeros(20)
for k in range(20):
    p = k*.005 +.85
    #startP = [p, 1-p]
    startP = np.array([.99, .01])
    transP = np.array([[p,1-p],[.01,.99]])
    emitP = np.array([[.7,.2,.1],[.1,.3,.6]])
    rmse = hmm_tester(observations, startP, transP, emitP)
    predictions = [rmse[l] for l in range(numStud) if not np.isnan(rmse[l])]
    error[k] = 1-np.average(predictions)
    
plt.scatter(np.arange(20),error)
"""