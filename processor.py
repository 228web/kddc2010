# -*- coding: utf-8 -*-
"""
Created on Thu Dec 03 21:12:13 2015

@author: John
"""

import numpy as np
import loader as ld
import tagger as tg
import ID_assigner as ida
import matplotlib.pyplot as plt

time_strings = ['Step Start Time','First Transaction Time',
                'Correct Transaction Time','Step End Time']
id_strings = ['Anon Student Id','Problem Name']

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
    
    #Scale
    datLen = len(xy_train[xy_keys[0]])
    
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
    dat_array[:,13] = xy_train['Step Duration (sec)']
    dat_array[:,14] = ld.check_final_answer(xy_train['Step Name'])
    dat_array[:,15] = np.array(xy_train['Row'],dtype=int)
    
    # Process Knowledge components
    tag_master = tg.string_tags(xy_train['KC(Default)'])

    # Process opportunity
    tag_array,opp_array = tg.tags_to_array(
                                xy_train['KC(Default)'],
                                xy_train['Opportunity(Default)'],
                                tag_master)
    
    
    return xy_keys, all_dicts, dat_array, tag_master, tag_array, opp_array
    
    
def processor_test(data, dicts, tags):
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
    tags : list
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
    dat_array = np.empty([datLen,7])
    dat_array[:,0] = xy_test['Anon Student Id']
    dat_array[:,1] = xy_test['Problem Name']
    dat_array[:,2:4] = xy_test['Problem Hierarchy']
    dat_array[:,4] = np.array(xy_test['Problem View'],dtype=int)
    dat_array[:,5] = ld.check_final_answer(xy_test['Step Name'])
    dat_array[:,6] = np.array(xy_test['Row'],dtype=int)
    
    # Process Knowledge components
    newTags = tg.string_tags(xy_test['KC(Default)'])
    tags.extend(newTags)

    # Process opportunity
    tag_array,opp_array = tg.tags_to_array(
                                xy_test['KC(Default)'],
                                xy_test['Opportunity(Default)'],
                                tags)
    
    
    return xy_keys, all_dicts, dat_array, tags, tag_array, opp_array