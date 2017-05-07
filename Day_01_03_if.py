# Day_01_03_if.py

# 문제
# 어떤 숫자가 홀수인지 짝수인지 알려주세요.
a = 3
if a%2 == 1:
    print('홀수')
else:
    print('짝수')

if a%2:
    print('홀수')
else:
    print('짝수')

a = 0
if a:
    print('홀수')
else:
    print('짝수')

# 문제
# 정수 : 음수, 0, 양수
a = 1
if a < 0:
    print('음수')
else:
    # print('제로, 양수')
    if a > 0:
        print('양수')
    else:
        print('제로')

if a < 0:
    print('음수')
elif a > 0:
    print('양수')
else:
    print('제로')

print('end of elif.')

# 문제
# 양수를 입력 받아서 자릿수를 알려주세요. (0 <= a < 1000)
# a = int(input())
a = 12

if a < 10:
    print(1)
if 10 <= a < 100:
    print(2)
if 100 <= a:
    print(3)

if a < 10:
    print(1)
elif a < 100:
    print(2)
else:
    print(3)
