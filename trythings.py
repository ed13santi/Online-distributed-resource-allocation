import numpy as np


s = [0,12,2,3,4]

mask = []
for i in range(5):
    if s[i] > 0:
        mask.append(1)
    else:
        mask.append(0)

a = None
for i, (switch, val) in enumerate(zip(mask, tmp)):
    if switch == 1:
        if a == None:
            a = i
        else:
            if val > tmp[a]:
                a = i

if a == None:
    a = len(self.taskTypes)

print(np.argmax(tmp))
