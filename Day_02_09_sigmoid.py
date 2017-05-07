# Day_02_09_sigmoid.py
import math


def sigmoid(z):
    return 1 / (1 + math.e ** -z)

print(sigmoid(-100))
print(sigmoid(-10))
print(sigmoid(0))
print(sigmoid(10))
print(sigmoid(100))

import matplotlib.pyplot as plt

xx, yy = [], []
for i in range(-10, 10):
    s = sigmoid(i)

    xx.append(i)
    yy.append(s)

# plt.plot(xx, yy, 'ro')
plt.plot(xx, yy)
plt.show()

# y=1, A
# y=0, B
# y*A + (1-y)*B




