# Day_03_03_softmax_python.py
import math

a, b, c = 2.0, 1.0, 0.1

total = math.e ** a + math.e ** b + math.e ** c

print((math.e ** a) / total)
print((math.e ** b) / total)
print((math.e ** c) / total)

# ---------------------- #

import numpy as np

a = np.arange(12).reshape(3, 4)
print(a)
print(a.shape)

print(np.sum(a))
print(np.sum(a, 0))     # column
print(np.sum(a, 1))     # row

