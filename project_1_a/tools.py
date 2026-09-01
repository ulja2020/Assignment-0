def lists(list_of_lists):
    """ 
    This function prints the elements of each list in the list and the sum of numerical values in each list

    Input
    -----
    list_of_lists (consists of several lists of numerical numbers and strings)

    Output
    -----
    The sum of all numerical values from all lists in the list (ignoring strings)
    """
    total_sum = 0

    for i in range(len(list_of_lists)):
        print(f"List {i+1} contains elements {list_of_lists[i]}")

        total = 0

        for x in list_of_lists[i]:
            if type(x) is int or type(x) is float:
                total = total + x

        print(f"The total sum in this list is {total}")

        total_sum = total_sum + total      # total sum of everything in all lists

    return total_sum

def dicts(dict_of_dicts):
    """
    Prints the elements of each dictionary in the dictionary and the sum of numerical values in each dictionary

    Input
    -----
    dict_of_dicts (consists of two dicts of numerical numbers and strings)

    Output
    -----
    The sum of all numerical values from all dicts in the dict (ignoring strings)
    """
    total_sum = 0

    for i in dict_of_dicts:
        print(f"{i} contains elements {dict_of_dicts[i]}")

        total = 0

        for x in dict_of_dicts[i]:
            if type(dict_of_dicts[i][x]) is int or type(dict_of_dicts[i][x]) is float:
                total = total + dict_of_dicts[i][x]

        print(f"The total sum in this dictionary is {total}")

        total_sum = total_sum + total      # total sum of everything in all dictionaries

    return total_sum


