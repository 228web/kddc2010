# -*- coding: utf-8 -*-
"""
Created on Sat Nov 07 13:34:10 2015

@author: John
"""

import numpy as np
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
    