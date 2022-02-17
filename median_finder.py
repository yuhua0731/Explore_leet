#!/usr/bin/env python3
import heapq

class MedianFinder:

    def __init__(self):
        self.size = 0
        self.upper_pq = [] # min heap
        self.lower_pq = [] # max heap

    def addNum(self, num: int) -> None:
        self.size += 1
        if self.size % 2:
            # total size is odd
            # get max element from lower, then push to upper
            heapq.heappush(self.upper_pq, -heapq.heappushpop(self.lower_pq, -num))
        else:
            # total size is even
            # get min element from upper, then push to lower
            heapq.heappush(self.lower_pq, -heapq.heappushpop(self.upper_pq, num))

    def findMedian(self) -> float:
        if self.size % 2:
            return self.upper_pq[0]
        else:
            return (self.upper_pq[0] + self.lower_pq[0]) / 2

if __name__ == '__main__':
    # Your MedianFinder object will be instantiated and called as such:
    obj = MedianFinder()
    obj.addNum(-1)
    print(obj.findMedian())
    obj.addNum(-2)
    print(obj.findMedian())
    obj.addNum(-3)
    print(obj.findMedian())
    obj.addNum(-4)
    print(obj.findMedian())
    obj.addNum(-5)
    print(obj.findMedian())