# Day_01_06_list.py

# collection : list, tuple, set, dict
#              []    ()     {}   {}

a = [1, 3, 5, 7, 9]
print(a)
print(a[0], a[1])

for i in range(len(a)):
    print(a[i], end=' ')
print()

print(a[len(a)-1], a[-1], a[-2])

a[0] = 99
print(a)

a.append(11)
print(a)

import random

random.seed(1)
print(random.randrange(100))
print(random.randrange(50, 100))
print(random.randrange(0, 100, 10))

# 문제
# 10개로 구성된 난수 리스트를 반환하는 함수를 만드세요.
def make_randoms():
    ns = []
    for _ in range(10):     # place holder
        ns.append(random.randrange(100))
    return ns

b = make_randoms()
print(b)

# 문제
# 리스트를 거꾸로 출력하는 2가지 코드를 만들어 보세요.
for i in range(len(b)):
    print(b[len(b)-1-i], end=' ')
print()

for i in range(-1, -len(b)-1, -1):
    print(b[i], end=' ')
print()

for i in range(1, len(b)+1):
    print(b[-i], end=' ')
print()

for i in reversed(range(len(b))):
    print(b[i], end=' ')
print()

for i in b:
    print(i, end=' ')
print()

for i in reversed(b):
    print(i, end=' ')
print()
print('-'*50)

# tuple : 리스트 상수 버전
a = (1, 3, 5)
print(a)
print(a[0], a[1], a[2])

# a[0] = 99
# a.append(11)

b1, b2 = 1, 2
print(b1, b2)
b3 = 1, 2       # packing
print(b3, b3[0], b3[1])
b4 = [5, 7]
print(b4, b4[0], b4[1])
b5, b6 = b4     # unpacking
print(b5, b6)
print('-'*50)

# 영한 : 영어 단어를 이용해서 한글 설명을 적어놓은 책
# 영어 : key
# 한글 : value

# d = {'name': 'kim', 'age': 20}
d = dict(name='kim', age=20)
print(d)
print(d['name'], d['age'])

d['money'] = 100        # insert
print(d)

d['money'] = 1000       # update
print(d)

for k in d:
    print(k, d[k])

print(d.items())

for i in d.items():
    print(i, i[0], i[1])

for k, v in d.items():
    print(k, v)

a1, a2 = 3, 5
e = {a1: 'hello', a2: 'python'}
print(e)














