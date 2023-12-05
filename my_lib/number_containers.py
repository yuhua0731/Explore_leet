#!/usr/bin/env python3
import collections
import heapq

class NumberContainers:

    def __init__(self):
        self.val = collections.defaultdict(int)
        self.con = collections.defaultdict(list)

    def change(self, index: int, number: int) -> None:
        self.val[index] = number
        heapq.heappush(self.con[number], index)

    def find(self, number: int) -> int:
        idx = self.con[number]
        while idx and self.val[idx[0]] != number:
            heapq.heappop(idx)
        return idx[0] if idx else -1


if __name__ == '__main__':
    # Your NumMatrix object will be instantiated and called as such:
    obj = NumberContainers()
    obj.change(2, 10)
    print(obj.find(10))
    print(obj.find(11))
    obj.change(1, 10)
    print(obj.find(10))