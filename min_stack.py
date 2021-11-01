#!/usr/bin/env python3
import heapq

class MinStack:

    def __init__(self):
        self.d = dict()
        self.pq = []  

    def push(self, val: int) -> None:
        self.d[len(self.d) + 1] = val
        heapq.heappush(self.pq, (val, len(self.d)))

    def pop(self) -> None:
        del self.d[len(self.d)]

    def top(self) -> int:
        return self.d[len(self.d)]

    def getMin(self) -> int:
        # in constant time conplexity
        while True:
            temp = self.pq[0]
            if temp[1] in self.d and self.d[temp[1]] == temp[0]: return temp[0]
            heapq.heappop(self.pq)


if __name__ == '__main__':
# Your MinStack object will be instantiated and called as such:
    obj = MinStack()
    obj.push(5)
    obj.push(-1)
    obj.push(2)
    obj.push(3)
    obj.pop()
    print(obj.top())
    print(obj.getMin())