#!/usr/bin/env python3
class kick:
    def trash_out(self, n, trash):
        ans = 0
        bin = list()
        for i in range(n):
            if trash[i] == '1': bin.append(i)
        for left, right in zip(bin, bin[1:]):
            print(left, right)
            middle = left + (right - left) // 2
            for i in range(left, middle + 1): ans += i - left
            for i in range(middle + 1, right): ans += right - i
        for i in range(bin[0]): ans += bin[0] - i
        for i in range(bin[-1], n): ans += i - bin[-1]
        return ans
if __name__ == '__main__':
    k = kick()
    print(k.trash_out(6, '100100'))

T = int(input())
for x in range(1, T + 1):
    N = int(input())
    S = str(input())
    ans = 0
    bin = list()
    for i in range(N):
        if S[i] == '1': bin.append(i)
    for left, right in zip(bin, bin[1:]):
        print(left, right)
        middle = left + (right - left) // 2
        for i in range(left, middle + 1): ans += i - left
        for i in range(middle + 1, right): ans += right - i
    for i in range(bin[0]): ans += bin[0] - i
    for i in range(bin[-1], N): ans += i - bin[-1]        
    print(f"Case #{x}: {ans}")