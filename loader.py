# -*- coding: utf-8 -*-
"""
Created on Sat Nov 07 13:34:10 2015

@author: John
"""

import numpy as np
import datetime as dt
from collections import defaultdict

trainDat = 'algebra_2005_2006_train.txt'
testDat = 'algebra_2005_2006_test.txt'

def loader(data):
    """
    Loads the data text file and parses
    
    Inputs
    ------
    data : string
        Text file from training or test data
    
    Returns
    -------
    keys : list
        list of strings for data
    x : dict
        dictionary of data
    """
    # Try opening the file
    try:
        with open(data, 'rb') as f:
            fileContent = f.read()
    except IOError:
        print "Error: Couldn't open file"
        return [],0
    
    # Split on newline character
    fileContent = fileContent.split('\n')
    
    # First line is the set of keys
    keys = fileContent[0].split('\t')
    content = fileContent[1:]
    
    # Initialize return dictionary
    keyLen = len(keys)
    numPts = len(content)
    x = {}
    #numPts = 2
    #keyLen = 2
    x = defaultdict(list)
    #for key in keys:
    #    x[key] = np.empty(numPts)
    #    print np.shape(x[key])
    
    print x
    for n in range(numPts-1):
        line = content[n].split('\t')
        if len(line) == keyLen:
            for k in range(keyLen):
                x[keys[k]].append(line[k])
        else:
            print 'Line length change on line '+str(n)
            print line
            
    return keys, x
    
def convert_times(time):
    """
    Converts strings in training data to datetime structures and then to
    seconds since Jan. 1, 1970.
    
    Inputs
    ------
    time : list
        list of Time events in training data
    
    Returns
    -------
    t : ndarray
        array of time events in seconds since Jan. 1, 1970. t[k] = 0 if entry
        doesn't match format, likely blank.
    """
    t0 = dt.datetime(1970,1,1)
    tLen = len(time)
    t = np.empty(tLen)
    for k in range(tLen):
        try:
            t1 = dt.datetime.strptime(time[k], '%Y-%m-%d %H:%M:%S.%f')
            deltaT = (t1-t0).total_seconds()
        except ValueError:
            print "Error, doesn't match format: %Y-%m-%d %H:%M:%S.%f"
            print 'Entry: '+str(k)+', Line: '+time[k]
            deltaT = 0.0
        
        t[k] = deltaT
        
    return t

def string_to_tag(data):
    """
    Convert strings to tag id numbers
    """
    
    return 0

def check_final_answer(stepName):
    """
    Check if string of 'Step Name' key is 'Final Answer' or not.
    
    Inputs
    ------
    stepName : list
        list of strings of Step Name
        
    Returns
    -------
    finalAns : ndarray
        1 if Final Answer, 0 otherwise
    """
    sNameLen = len(stepName)
    finalAns = np.empty(sNameLen)
    
    for k in range(sNameLen):
        if 'Final' in stepName[k]:
            finalAns[k] = 1
        else:
            finalAns[k] = 0
            
    return finalAns

    