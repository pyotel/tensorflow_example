# Day_01_02_operator.py

# ctrl + shift + f10

# 연산 : 산술, 관계, 논리
# 산술 : +  -  *  /  //  **  %
a, b = 7, 3
print(a + b)
print(a - b)
print(a * b)
print(a / b)        # 실수 나눗셈
print(a // b)       # 몫, 정수 나눗셈
print(a ** b)       # 지수
print(a % b)        # 나머지
print('-'*50)

# 관계 : >  >=  <  <=  ==  !=
print(a > b)
print(a >= b)
print(a < b)
print(a <= b)
print(a == b)
print(a != b)

c = a != b
# (3 + 4) * 5
print(int(c))
print(int(a != b))
print('-'*20 + '='*20)

# 논리 : and  or  not
print(True and True)
print(True and False)
print(False and True)
print(False and False)
print('-'*50)

# 문제
# 나이를 입력 받아서 10대이면 True, 아니면 False를 출력해 보세요.
# (10 ~ 19)
age = input()
print(type(age))
age = int(age)

# print(age >= 10 and age < 20)
print(10 <= age < 20)           # 범위 연산
