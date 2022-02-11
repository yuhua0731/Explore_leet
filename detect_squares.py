#!/usr/bin/env python3
from typing import List
import collections
import itertools

class DetectSquares:

    def __init__(self):
        # we need three dict
        # dict 1: key = point(i, j) value = amount
        # dict 2: key = x index | value = dict 4: key = y index | value = amount
        # dict 3: key = y index | value = dict 5: key = x index | value = amount
        self.node = collections.defaultdict(int)
        self.same_x = collections.defaultdict(dict)
        self.same_y = collections.defaultdict(dict)

    def add(self, point: List[int]) -> None:
        x, y = point[0], point[1]
        self.node[(x, y)] += 1
        if y not in self.same_x[y]: self.same_x[x][y] = 0
        self.same_x[x][y] += 1
        if x not in self.same_y[x]: self.same_y[y][x] = 0
        self.same_y[y][x] += 1

    def count(self, point: List[int]) -> int:
        # we have x, y of point
        # with x, we want to fetch the list of nodes which share the same x, while y is different, store their y in y_arr
        # with y, we want to fetch the list of nodes which share the same y, while x is different, store their x in x_arr
        # for i, j in product(x_arr, y_arr), we check if [i, j] exist in our points
        # if yes, we add the amount of squares to the answer
        ans = 0
        x, y = point[0], point[1]
        y_dict, x_dict = self.same_x[x], self.same_y[y]
        for i, j in itertools.product(x_dict.keys(), y_dict.keys()):
            if i == x or j == y: continue
            ans += x_dict[i] * y_dict[j] * self.node[(i, j)]
        return ans

if __name__ == '__main__':
    # Your DetectSquares object will be instantiated and called as such:
    obj = DetectSquares()
    obj.add([3, 10])
    obj.add([11, 2])
    obj.add([3, 2])
    print(obj.count([11, 10]))