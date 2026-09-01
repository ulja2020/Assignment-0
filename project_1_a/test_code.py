from tools import lists
from tools import dicts

"""defining object a list of list and printing the result from tools.py"""
my_list = [[1, "nvfkv", "mn"], [36, 2], [10, "hello", 4]]

sum_result = lists(my_list)
assert sum_result == 53          #checking that the total sum of everything in all lists is 53
print(sum_result)



"""defining object a dict of dict and printing the result from tools.py"""
my_dict = {
    "dict1": {"a": 10, "b": 3, "c": "mn"},
    "dict2": {"a": 5, "b": 2}
}
sum_result = dicts(my_dict)
assert sum_result == 20            #checking that the total sum of everything in all dictionaries is 20
print(sum_result)
