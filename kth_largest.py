#!/usr/bin/env python3
from typing import List
import heapq

class KthLargest:
    """
    a class to find the kth largest element in a list
    min-heap
    """
    def __init__(self, k: int, nums: List[int]):
        self.pq = [] # heapq to store the k largest elements
        nums.append(- 10 ** 4 - 1)
        for i in nums:
            if len(self.pq) == k:
                heapq.heappushpop(self.pq, i)
            else:
                heapq.heappush(self.pq, i)

    def add(self, val: int) -> int:
        heapq.heappushpop(self.pq, val)
        return self.pq[0]

if __name__ == '__main__':
    # Your KthLargest object will be instantiated and called as such:
    obj = KthLargest(3, [4,5,8,2])
    print(obj.add(3))