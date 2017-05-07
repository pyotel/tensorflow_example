# Day_02_02_slicing.py

a = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]

print(a[0], a[-1])
print(a[3:7])           # 시작, 종료
print(a[0:len(a)//2])
print(a[len(a)//2:len(a)])
print(a[:len(a)//2])
print(a[len(a)//2:])

# 문제
# 짝수 번째만 출력해 보세요.
# 홀수 번째만 출력해 보세요.
# 거꾸로 출력해 보세요.
print(a[::])
print(a[::2])
print(a[1::2])
print(a[3:4])
print(a[3:3])
print(a[len(a)-1:0:-1])
print(a[-1:0:-1])
print(a[-1:-1:-1])
print(a[-1::-1])
print(a[::-1])
