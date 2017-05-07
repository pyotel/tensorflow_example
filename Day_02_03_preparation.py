# Day_02_03_preparation.py
import numpy as np
import matplotlib.pyplot as plt


def not_used():
    x = [1, 2, 3, 4, 5]
    y = [1, 3, 5, 7, 9]

    plt.plot(x, y)
    plt.plot(x, y, 'ro')
    plt.show()


# y = ax + b
# y = 2x + 3
# def equation(x):)))
#     y = []
#     for i in x:
#         y.append(2*i + 3)
#     return y


def equation(x, a, b):
    y = []
    for i in x:
        y.append(a*i + b)
    return y


def distance(label, hypo):
    dist = 0
    for i in range(len(label)):
        # if label[i] > hypo[i]:
        #     dist += label[i] - hypo[i]
        # else:
        #     dist += hypo[i] - label[i]
        dist += abs(hypo[i] - label[i])

    return dist


x  = [1, 2, 3, 4, 5]
y  = equation(x, 2, 3)      # y = 2x + 3
y1 = equation(x, 3, -1)     # y = 3x - 1
y2 = equation(x, 3, 1)      # y = 3x + 1
y3 = equation(x, 2, 5)      # y = 2x + 5

# 문제
# equation을 올바로 작성해 보세요.

# 문제
# distance 함수를 채워 보세요.

print(distance(y, y1))
print(distance(y, y2))
print(distance(y, y3))

plt.plot(x, y, 'ro')
plt.plot(x, y, 'y')
plt.plot(x, y1, 'r')
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'b')
plt.show()




