# Day_02_04_cost.py
import matplotlib.pyplot as plt

def cost(y, x, a, b):
    c = 0
    for i in range(len(y)):
        yhat = a * x[i] + b
        c += (y[i] - yhat) ** 2
    return c/len(y)


def gradient_descent(y, x, a):
    c = 0
    for i in range(len(y)):
        # c += (a*x[i] - y[i]) ** 2
        c += 2 * (a * x[i] - y[i]) * x[i]
    return c/len(y)


# a = 1, b = 0
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

xx, yy = [], []
for i in range(-30, 50, 1):
    # print(i/10)
    c = cost(y, x, i/10, 0)
    print(i/10, c)

    xx.append(i/10)
    yy.append(c)

# plt.plot(xx, yy, 'ro')
# plt.show()

# 미분 : 순간변화량

# y = x         1
# y = 2x        2
# y = 3         0
# y = x^2       2x
# y = (x-3)^2   2(x-3)
print('-'*50)

a = -10
for i in range(100):
    c = cost(y, x, a, 0)
    grad = gradient_descent(y, x, a)
    a -= grad * 0.01

    print('{:.6f} : {:.6f} {}'.format(a, grad, c))


p1 = 20
print(a*p1)






