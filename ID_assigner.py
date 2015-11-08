def student_ID_assigner(student_IDs):
    """
    Converts student IDs to integers, stores IDs in separate dictionary
    
    Inputs
    ------
    Student IDs : list
        list of strings of anonymous student IDs
        
    Returns
    -------
    student IDs : list
        The same list, but with integer IDs instead of strings

    ID dict : dictionary
	The student IDs in a dictionary for quick lookup.

    Usage: [stud_ID,stud_dict] = student_ID_assigner(xy_train['Anon Student Id']
    """
    IDstr_Len = len(student_IDs)
    ID_list = np.empty(IDstr_Len,dtype=int)
    ID_dict = defaultdict(list)
    dict_len = 0

#    print IDstr_Len

    for k in range(IDstr_Len):
        curr_ID = str(student_IDs[k])
        if curr_ID not in ID_dict:
            ID_dict.update({curr_ID: dict_len})
            ID_list[k] = dict_len
            dict_len = dict_len + 1
        else:
            ID_list[k] = ID_dict.get(curr_ID)


    return ID_list, ID_dict
