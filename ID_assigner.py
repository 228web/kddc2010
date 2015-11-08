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
