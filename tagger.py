# -*- coding: utf-8 -*-
"""
Created on Sun Nov 08 12:22:19 2015

@author: John
"""

import numpy as np

def string_tags(KCList):
    """
    Takes list of 'KC(Default)' dictionary string tags and creates a master of 
    list of all tags.
    
    Inputs
    ------
    KCList : list
        list of 'KC(Default)' string tags
        
    Returns
    -------
    tags : list
        list of string tags without duplicates and split on '~~'
    """
    listLen = len(KCList)
    tags = []
    for k in range(listLen):
        # split on '~~'
        tagsK = KCList[k].split('~~')
        
        # check if a single string first
        if isinstance(tagsK, basestring):
            # check if already included as a tag, if not then append
            if tagsK not in tags:
                tags.append(tagsK)
        else:
            # check through all strings if included as tag, if not then append
            for l in range(len(tagsK)):
                if tagsK[l] not in tags:
                    tags.append(tagsK[l])
        
    return tags
    
def tags_to_array(KCList, oppList, tagList):
    """
    Takes the list of 'KC(Default)' string tags and master list of all tags and
    creates sparse array representing present tags.
    
    Inputs
    ------
    KCList : list
        list of all 'KC(Default)' string tags
    oppList : list
        list of all 'Opportunity' int tags
    tagList : list
        list of string tags without duplicates and split on '~~'
        
    Returns
    -------
    tagArray : ndarray
        binary sparse array of present tags
    """
    KCLen = len(KCList)
    tagLen = len(tagList)
    tagArray = np.zeros([KCLen,tagLen])
    oppArray = np.zeros([KCLen,tagLen])
    index = 0
    
    for k in range(KCLen):
        # split on '~~'
        tagsK = KCList[k].split('~~')
        oppK = oppList[k].split('~~')
        
        #check if a single string first
        if isinstance(tagsK, basestring):
            # check which index tag is in
            index = tagList.index(tagsK)
            tagArray[k,index] = 1
            if oppK == '':
                oppArray[k,index] = -1
            else:
                oppArray[k,index] = oppK
        else:
            for l in range(len(tagsK)):
                index = tagList.index(tagsK[l])
                tagArray[k,index] = 1
                if oppK[l] == '':
                    oppArray[k,index] = -1
                else:
                    oppArray[k,index] = oppK[l]
        
    return tagArray, oppArray
    