# -*- coding: utf-8 -*-
"""
Created on Thu Dec 03 21:12:13 2015

@author: John
"""

import numpy as np
import loader as ld
import tagger as tg
import ID_assigner as ida
import hmm as hmm
import matplotlib.pyplot as plt

time_strings = ['Step Start Time','First Transaction Time',
                'Correct Transaction Time','Step End Time']
id_strings = ['Anon Student Id','Problem Name']
dur_strings = ['Step Duration (sec)','Correct Step Duration (sec)',
               'Error Step Duration (sec)']

def processor_train(data):
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
    
    nsize = len(xy_train['Anon Student Id'])
    
    # Process time strings to seconds
    for i in range(4):
        print 'Processing ' + time_strings[i]
        xy_train[time_strings[i]] = ld.convert_times(xy_train[time_strings[i]])
        
    # Convert durations to seconds, 0 if not present    
    for i in range(len(dur_strings)):
        print 'Processing '+dur_strings[i]
        for j in range(nsize):
            if xy_train[dur_strings[i]][j] == '':
                xy_train[dur_strings[i]][j] = 0.0
            else:
                xy_train[dur_strings[i]][j] = float(xy_train[dur_strings[i]][j])

    # Dictionary of anonId and problem tags
    all_dicts = []

    # Process string ids
    for i in range(2):
        print 'Processing ' + id_strings[i]
        xy_train[id_strings[i]],temp = ida.ID_assigner(xy_train[id_strings[i]])
        all_dicts.append(temp)

    print 'Processing Problem Hierarchy'
    xy_train['Problem Hierarchy'],temp,temp2 = ida.unit_ID_assigner(
                                                xy_train['Problem Hierarchy'])
    all_dicts.append(temp)
    all_dicts.append(temp2)
    
    #Scale
    datLen = len(xy_train[xy_keys[0]])
    
    print 'Turning things to numpy array'
    #These are the variables I care about at the moment, add if want more - JGL
    # 'Anon Student Id','Incorrects','Corrects','Problem View',
    #'Correct Transaction Time','Correct First Attempt','Step Start Time',
    #'First Transaction Time','Problem Hierarchy','Hints','Step End Time']
    # KC(Default) and Opportunity(Default) separate arrays.
    dat_array = np.empty([datLen,16])
    dat_array[:,0] = xy_train['Anon Student Id']
    dat_array[:,1] = xy_train['Problem Name']
    dat_array[:,2:4] = xy_train['Problem Hierarchy']
    dat_array[:,4] = np.array(xy_train['Incorrects'],dtype=int)
    dat_array[:,5] = np.array(xy_train['Hints'],dtype=int)
    dat_array[:,6] = np.array(xy_train['Corrects'],dtype=int)
    dat_array[:,7] = np.array(xy_train['Correct First Attempt'],dtype=int)
    dat_array[:,8] = np.array(xy_train['Problem View'],dtype=int)
    dat_array[:,9] = xy_train['Step Start Time']
    dat_array[:,10] = xy_train['First Transaction Time']
    dat_array[:,11] = xy_train['Correct Transaction Time']
    dat_array[:,12] = xy_train['Step End Time']
    dat_array[:,13] = np.array(xy_train['Step Duration (sec)'])
    dat_array[:,14] = ld.check_final_answer(xy_train['Step Name'])
    dat_array[:,15] = np.array(xy_train['Row'],dtype=int)
    
    print 'Processing KC(Default)'
    # Process Knowledge components
    tag_master = tg.string_tags(xy_train['KC(Default)'])

    # Process opportunity
    tag_array,opp_array = tg.tags_to_array(
                                xy_train['KC(Default)'],
                                xy_train['Opportunity(Default)'],
                                tag_master)
    
    
    return xy_keys, all_dicts, dat_array, tag_master, tag_array, opp_array
    
    
def processor_test(data, dicts, tags, master = False):
    """
    This is a functional form of data_processor sans some features.
    
    Inputs
    ------
    data : string
        path and file name of data (e.g. ld.testDat)
    dicts : list
        list of dictionaries from processing training data
    tags : list
        list of tags from processing training data
        
    Returns
    -------
    xy_keys : list
        list of keys for data dictionary
    dat_array : ndarray
        parsed data dictionary now stored in numpy array
    tags2 : list
        list of tags from Knowledge Component data
    tag_array : ndarray
        array of knowledge component presence in each question
    opp_array : ndarray
        array of opportunity count for each component in each question
    """
    xy_keys,xy_test = ld.loader(data)

    # Dictionary of anonId and problem tags
    all_dicts = []

    # Process string ids
    for i in range(2):
        print 'Processing ' + id_strings[i]
        xy_test[id_strings[i]],temp = ida.ID_assigner_TEST(xy_test[id_strings[i]],
                                        dicts[i])
        all_dicts.append(temp)

    xy_test['Problem Hierarchy'],temp,temp2 = ida.unit_ID_assigner_TEST(
                                                xy_test['Problem Hierarchy'],
                                                dicts[2],dicts[3])
    all_dicts.append(temp)
    all_dicts.append(temp2)
    
    #Scale
    datLen = len(xy_test[xy_keys[0]])
    
    #These are the variables I care about at the moment, add if want more - JGL
    # 'Anon Student Id','Incorrects','Corrects','Problem View',
    #'Correct Transaction Time','Correct First Attempt','Step Start Time',
    #'First Transaction Time','Problem Hierarchy','Hints','Step End Time']
    # KC(Default) and Opportunity(Default) separate arrays.
    dat_array = np.zeros([datLen,16])
    dat_array[:,0] = xy_test['Anon Student Id']
    dat_array[:,1] = xy_test['Problem Name']
    dat_array[:,2:4] = xy_test['Problem Hierarchy']
    dat_array[:,8] = np.array(xy_test['Problem View'],dtype=int)
    dat_array[:,14] = ld.check_final_answer(xy_test['Step Name'])
    dat_array[:,15] = np.array(xy_test['Row'],dtype=int)
    
    if master == True:
        nsize = len(xy_test['Anon Student Id'])
        
        # Process time strings to seconds
        for i in range(4):
            print 'Processing ' + time_strings[i]
            xy_test[time_strings[i]] = ld.convert_times(xy_test[time_strings[i]])
            
        # Convert durations to seconds, 0 if not present    
        for i in range(len(dur_strings)):
            print 'Processing '+dur_strings[i]
            for j in range(nsize):
                if xy_test[dur_strings[i]][j] == '':
                    xy_test[dur_strings[i]][j] = 0.0
                else:
                    xy_test[dur_strings[i]][j] = float(xy_test[dur_strings[i]][j])
                
        dat_array[:,4] = np.array(xy_test['Incorrects'],dtype=int)
        dat_array[:,5] = np.array(xy_test['Hints'],dtype=int)
        dat_array[:,6] = np.array(xy_test['Corrects'],dtype=int)
        dat_array[:,7] = np.array(xy_test['Correct First Attempt'],dtype=int)
        dat_array[:,8] = np.array(xy_test['Problem View'],dtype=int)
        dat_array[:,9] = xy_test['Step Start Time']
        dat_array[:,10] = xy_test['First Transaction Time']
        dat_array[:,11] = xy_test['Correct Transaction Time']
        dat_array[:,12] = xy_test['Step End Time']
        dat_array[:,13] = np.array(xy_test['Step Duration (sec)'])
        
    # Process Knowledge components
    newTags = tg.string_tags(xy_test['KC(Default)'])
    
    #Better to make a copy of tags
    tags2 = list(tags)
    tags2.extend(newTags)

    # Process opportunity
    tag_array,opp_array = tg.tags_to_array(
                                xy_test['KC(Default)'],
                                xy_test['Opportunity(Default)'],
                                tags2)
    
    
    return xy_keys, all_dicts, dat_array, tags2, tag_array, opp_array
    
    
def stud_splits(studId):
    """
    Finds what indices mark a change in student along with the point 0 to start
    
    Inputs
    ------
    studId : ndarray
        student data, if from processor_train, is data[:,0]
        
    Returns
    -------
    splitIds : list
        list of indices marking a new student in data, includes point 0
    """
    splitId1 = [k+1 for k in range(len(studId)-1) if studId[k+1] != studId[k]]
    splitIds = [0]
    splitIds.extend(splitId1)
    return splitIds
    
    
def row_stud_id(test, train, studSplitIds):
    """
    For each test point, returns the index pair of where to start and end for 
    the hmm predictor. Assumes split indices came from stud_splits so include
    zero but not last point index. Also, assumes data comes from process_(train,
    test).
    """
    testLen = len(test)
    numStud = len(studSplitIds)
    outPairs = np.zeros([testLen,2])
    for k in range(testLen):
        num = test[k,0]-1
        if num < numStud:
            if num == numStud-1:
                print int(num)
                outPairs[k,0] = studSplitIds[-1]
                outPairs[k,1] = np.min([test[k,-1]-1,train[-1,-1]])
                #Need to make sure don't try to handle a point from before
                if outPairs[k,0]<0 and outPairs[k,1]<0:
                    outPairs[k] = np.zeros(2)
                    
            else:
                outPairs[k,0] = studSplitIds[int(num)]
                outPairs[k,1] = np.min([test[k,-1]-
                                    train[studSplitIds[int(num)],-1],
                                    train[studSplitIds[int(num)+1],-1]])
                if outPairs[k,0]<0 or outPairs[k,1]<0:
                    outPairs[k] = np.zeros(2)
        else:
            print num, k
            
    return outPairs
    
    
def test_MCSGDBW(train, test, splitIds, outPairs, sp, tp, ep,index, binList):
    testLen = len(test)
    #binList = [0,.1,1.1,6,15,30,100]
    bins = np.array(binList)
    binVec = bins#(bins[1:]-bins[:-1])/2.+bins[:-1]
    observations = np.digitize(train[:,index],binList)
    observations -= 1.
    print np.max(observations),np.min(observations)
    for k in range(testLen):
        if outPairs[k,1] != 0:
            print k
            index0 = outPairs[k,0]
            index1 = outPairs[k,1]+outPairs[k,0]
            f,b,pF,pB,post = hmm.frwd_bkwd(observations[index0:index1], 
                                           sp, tp, ep)
            state = np.dot(ep.T,np.dot(tp.T,post[-1]))
            test[k,index] = np.dot(binVec,state)
            print test[k,index]
        else:
            test[k,index] = 0
            
    test[:,13] = np.digitize(test[:,index],binList)
    
    return test
    

def recall(master, pred):
    tpfn = np.sum(master[:,7])
    tp = [pred[k] for k in range(len(pred)) if master[k,7]==1]
    tp = np.sum(tp)
    print tp, tpfn
    return tp/float(tpfn)
    
def precision(master, pred):
    tpfp = np.sum(pred)
    tp = [pred[k] for k in range(len(pred)) if master[k,7]==1]
    tp = np.sum(tp)
    print tp, tpfp
    return tp/float(tpfp)