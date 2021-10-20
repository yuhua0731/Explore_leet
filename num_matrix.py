#!/usr/bin/env python3
from typing import List

class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.m = matrix
        m, n = len(matrix), len(matrix[0])
        for i in range(m):
            for j in range(n):
                self.m[i][j] += (self.m[i - 1][j] if i > 0 else 0) + (self.m[i][j - 1] if j > 0 else 0) - (self.m[i - 1][j - 1] if i > 0 and j > 0 else 0)
        print(self.m)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.m[row2][col2] - (self.m[row1 - 1][col2] if row1 > 0 else 0) - (self.m[row2][col1 - 1] if col1 > 0 else 0) + (self.m[row1 - 1][col1 - 1] if row1 > 0 and col1 > 0 else 0)

if __name__ == '__main__':
    # Your NumMatrix object will be instantiated and called as such:
    obj = NumMatrix([[67,64,78],[99,98,38],[82,46,46],[6,52,55],[55,99,45]])
    print(obj.sumRegion(0, 1, 4, 2))