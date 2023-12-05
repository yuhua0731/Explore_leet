#!/usr/bin/env python3
import heapq

class SmallestInfiniteSet:
    
    def __init__(self):
        self.num = [True] * 1000
        self.s = [i + 1 for i in range(1000)]
        heapq.heapify(self.s)

    def popSmallest(self) -> int:
        while self.s and not self.num[self.s[0] - 1]:
            heapq.heappop(self.s)
        ret = heapq.heappop(self.s)
        self.num[ret - 1] = False
        return ret

    def addBack(self, num: int) -> None:
        if self.num[num - 1]: return
        self.num[num - 1] = True
        heapq.heappush(self.s, num)


if __name__ == '__main__':
    # Your SmallestInfiniteSet object will be instantiated and called as such:
    obj = SmallestInfiniteSet()
    param_1 = obj.popSmallest()
    param_1 = obj.popSmallest()
    obj.addBack(1)
    param_1 = obj.popSmallest()