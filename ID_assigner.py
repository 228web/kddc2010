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
    dict_len = 0

#    print IDstr_Len

    for k in range(IDstr_Len):
        curr_ID = str(raw_IDs[k])
        if curr_ID not in ID_dict:
            ID_dict.update({curr_ID: dict_len})
            ID_list[k] = dict_len
            dict_len = dict_len + 1
        else:
            ID_list[k] = ID_dict.get(curr_ID)


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
    unit_dlen = 0
    sect_dlen = 0

#    print IDstr_Len

    for k in range(IDstr_Len):
        curr_ID = str(raw_IDs[k]).split(', ')
        if curr_ID[0] not in unit_dict:
            unit_dict.update({curr_ID[0]: unit_dlen})
            ID_list[k,0] = unit_dlen
            unit_dlen = unit_dlen + 1
        else:
            ID_list[k,0] = unit_dict.get(curr_ID[0])

        if curr_ID[1] not in sect_dict:
            sect_dict.update({curr_ID[1]: sect_dlen})
            ID_list[k,1] = sect_dlen
            sect_dlen = sect_dlen + 1
        else:
            ID_list[k,1] = sect_dict.get(curr_ID[1])

    return ID_list,unit_dict,sect_dict
