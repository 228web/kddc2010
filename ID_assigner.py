#!/usr/bin/python
import numpy as np
from collections import defaultdict

def ID_assigner(raw_IDs):
    """
    Converts the IDs to integers, stores in in separate dictionary
    
    Inputs
    ------
    raw IDs : list
        list of strings of IDs
        
    Returns
    -------
    filtered IDs : list
        The same list, but with integer IDs instead of strings

    ID dict : dictionary
	The raw IDs in a dictionary for quick lookup.

    Usage: [filt_ID,filt_dict] = ID_assigner(xy_train[STRING HERE])
    Keys that need filtering: 'Anon Student Id','Problem Name'
    """
    IDstr_Len = len(raw_IDs)
    ID_list = np.empty(IDstr_Len,dtype=int)
    ID_dict = defaultdict(list)

    dict_len = 1
    ID_dict.update({0 : ''})
#    print IDstr_Len

    for k in range(IDstr_Len):
        curr_ID = str(raw_IDs[k])
        if curr_ID == '':
            ID_list[k] = 0
        elif curr_ID not in ID_dict.values():
            ID_dict.update({dict_len: curr_ID})
            ID_list[k] = dict_len
            dict_len = dict_len + 1
        else:
            ID_list[k] = ID_dict.values().index(curr_ID)


    return ID_list, ID_dict



def unit_ID_assigner(raw_IDs):
    """
    Splits the problem hierarchy IDs into unuits and sections, assigns integers and stores in in separate dictionary
    
    Inputs
    ------
    raw IDs : list
        list of strings of IDs
        
    Returns
    -------
    filtered IDs : list
        Unit and Section list, but with integer IDs instead of strings

    ID dict : dictionary
	The raw IDs in a dictionary for quick lookup.

    Usage: [hier_ID,unit_dict,sect_dict] = ID_assigner(xy_train['Problem Hierarchy'])
    This function works under the assumption that 'Problem Hierarchy' strings
    are in the form 'Unit XXX, Section XXX'
    """
    IDstr_Len = len(raw_IDs)
    ID_list = np.empty((IDstr_Len,2),dtype=int)
    unit_dict = defaultdict(list)
    sect_dict = defaultdict(list)

    unit_dlen = 1
    sect_dlen = 1
    unit_dict.update({0 : ''})
    sect_dict.update({0 : ''})

#    print IDstr_Len

    for k in range(IDstr_Len):
        curr_ID = str(raw_IDs[k]).split(', ')

        if curr_ID[0] == '':
            unit_list[k] = 0
        elif curr_ID[0] not in unit_dict.values():
            unit_dict.update({unit_dlen: curr_ID[0]})
            ID_list[k,0] = unit_dlen
            unit_dlen = unit_dlen + 1
        else:
            ID_list[k,0] = unit_dict.values().index(curr_ID[0])

        if curr_ID[1] == '':
            sect_list[k] = 0
        elif curr_ID[1] not in sect_dict.values():
            sect_dict.update({sect_dlen: curr_ID[1]})
            ID_list[k,1] = sect_dlen
            sect_dlen = sect_dlen + 1
        else:
            ID_list[k,1] = sect_dict.values().index(curr_ID[1])

    return ID_list,unit_dict,sect_dict







def ID_assigner_TEST(raw_IDs,iDict):
    """
    Converts the IDs to integers, using dictionary
    
    Inputs
    ------
    raw IDs : list
        list of strings of IDs
    idict : dictionary
	The dictionary of student IDs calculated from the training data

    Returns
    -------
    filtered IDs : list
        The same list, but with integer IDs instead of strings


    Usage: filt_ID = ID_assigner(xy_train[STRING HERE],iDict)
    Keys that need filtering: 'Anon Student Id','Problem Name'
    """
    IDstr_Len = len(raw_IDs)
    dict_len = len(iDict)

    ID_list = np.empty(IDstr_Len,dtype=int)

    for k in range(IDstr_Len):
        curr_ID = str(raw_IDs[k])
        if curr_ID == '':
            ID_list[k] = 0
        elif curr_ID not in iDict.values():
            dict_len = dict_len + 1
            iDict.update({dict_len: curr_ID})
            ID_list[k] = dict_len
        else:
            ID_list[k] = iDict.values().index(curr_ID)

    return ID_list, iDict



def unit_ID_assigner_TEST(raw_IDs,uDict,sDict):
    """
    Processes the problem hierarchy using a given dictionary
    
    Inputs
    ------
    raw IDs : list
        list of strings of IDs

    uDict : dictionary
	The raw IDs for the units
    sDict : dictionary
	The raw IDs for the sections


    Returns
    -------
    filtered IDs : list
        Unit and Section list, but with integer IDs instead of strings


    Usage: hier_ID = ID_assigner(xy_train['Problem Hierarchy'],uDict,sDict)
    This function works under the assumption that 'Problem Hierarchy' strings
    are in the form 'Unit XXX, Section XXX'
    """
    IDstr_Len = len(raw_IDs)
    ID_list = np.empty((IDstr_Len,2),dtype=int)

    uLen = len(uDict)
    sLen = len(sDict)


    for k in range(IDstr_Len):
        curr_ID = str(raw_IDs[k]).split(', ')

        if curr_ID[0] == '':
            unit_list[k] = 0
        elif curr_ID[0] not in uDict.values():
            uLen = uLen + 1
            uDict.update({uLen: curr_ID[0]})
            ID_list[k,0] = uLen
        else:
            ID_list[k,0] = uDict.values().index(curr_ID[0])

        if curr_ID[1] == '':
            sect_list[k] = 0
        elif curr_ID[1] not in sDict.values():
            sLen = sLen + 1
            sDict.update({sLen: curr_ID[1]})
            ID_list[k,1] = sLen
        else:
            ID_list[k,1] = sDict.values().index(curr_ID[1])

    return ID_list,uDict,sDict
