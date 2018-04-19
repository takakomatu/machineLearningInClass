# -*- coding: utf-8 -*-
import numpy as np
labels=[2,0,0,0,1,2,2,1,2,2,1]
s=set(labels)
print(set(labels))
#set() creates a set obeject set cant hold same value


labels=["a","b","a","c","a","b","a","c","a","a"]
print(np.array(labels) == np.array([True, True, True, True, True, False,False,False,False,False]))
print(set(labels)) # set
print(sorted(set(labels))) # list
#self.classes = sorted(set(self.labels))
print(labels)
