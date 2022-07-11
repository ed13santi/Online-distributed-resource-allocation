import numpy as np

def init_nested_dict(keys, val):
    tmp = val
    for k in np.flip(keys):
        tmp = {k: tmp}
    return tmp

def modify_nested_dict(d, keys, val, i):
    tmp = init_nested_dict(keys[i:], val) 
    tmp_dict = d
    for k in keys[:i]:
        if k not in tmp_dict:
            tmp_dict[k] = {}
        tmp_dict = tmp_dict[k]
    tmp_dict[keys[i]] = tmp[keys[i]]
    return d

d = {}
keys = [0]
val = {}
i = 0

out = modify_nested_dict({}, [0], {}, 0)
print(out)

out = modify_nested_dict(out, [5], {}, 0)
print(out)

out = modify_nested_dict(out, [5, 3], {}, 1)
print(out)

out = modify_nested_dict(out, [0, 3], {}, 1)
print(out)

out = modify_nested_dict(out, [2, 3], {}, 1)
print(out)