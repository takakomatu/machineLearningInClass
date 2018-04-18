from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

r = .25 # growth rate / yr
K = 100 # carrying capacity
t = 25 # number of years
num = np.zeros(t+1)
num[0] = 1
for i in range(t):
    num[i+1] = num[i]+r*num[i]*(K-num[i]/K) 
plt.plot(range(t+1),num, 'b')
plt.xlabel('')
plt.ylabel('')
plt.title('Growth rate: 0.25, Carrying Capacity = 100')
plt.axvline(np.argmax(np.diff(num)),  color = 'k'  )
plt.show()