# Day_01_05_function.py

def not_used():
    # 함수 : 어떤 기능을 수행하는 코드 영역
    # 교수  --> 데이터 -->  나 : 매개변수
    # 교수  <-- 데이터 <--  나 : 반환값

    # 매개변수 없고, 반환값 없고.
    def f_1():
        print('hello')

    f_1()

    # 매개변수 있고, 반환값 없고.
    def f_2(a, b):
        print('f_2', a, b)

    f_2(12, 34)

    # 매개변수 없고, 반환값 있고.
    def f_3():
        return 7

    # a = return 7
    a = f_3()
    print(a)
    print(f_3())

    # 매개변수 있고, 반환값 있고.
    # 두 개의 정수 중에서 큰 값을 반환하는 함수를 만드세요.
    def f_4(a, b):
        # if a > b:
        #     return a
        # else:
        #     return b

        # if a > b:
        #     return a
        # return b

        if a > b:
            b = a
        return b

    print(f_4(3, 9))
    print(f_4(9, 3))


def order(a, b):
    if a > b:
        return a, b
    return b, a

a = order(3, 7)
print(a, a[0], a[1])

big, small = order(3, 7)
print(big, small)


def f_5(*args):     # 가변인자
    print(args)

f_5()
f_5(1)
f_5(1, 2)
f_5(1, 2, 3)


def f_6(a, b, c=0):     # default argument
    print(a, b, c, sep='*')

f_6(4, 5, 6)            # position argument
f_6(4, 5)
f_6(a=4, b=5, c=6)      # keyword argument
f_6(c=6, a=4, b=5)


