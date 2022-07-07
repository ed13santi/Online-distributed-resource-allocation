import numpy as np

def init_nested_dict(keys, val):
    tmp = val
    for k in np.flip(keys):
        tmp = {k: tmp}
    return tmp

def modify_nested_dict(d, keys, val, i):
    tmp = init_nested_dict(keys[i:], val) #initialise Q value at 0
    for j in range(i+1, 0, -1):
        tmp_dict = d
        for k in keys[:-1]:
            tmp_dict = tmp_dict[k]
        tmp_dict[i] = tmp
        tmp = tmp_dict
    
    return tmp

d = {}
keys = [0]
val = {}
i = 0

out = modify_nested_dict({}, [0], {}, 0)
print(out)


out = modify_nested_dict(out, [5], {}, 0)
print(out)