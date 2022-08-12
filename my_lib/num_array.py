#!/usr/bin/env python3
from typing import List

class NumArray:

    def __init__(self, nums: List[int]):
        self.pre_sum = [0]
        for i in nums:
            self.pre_sum.append(self.pre_sum[-1] + i)

    def sumRange(self, left: int, right: int) -> int:
        return self.pre_sum[right + 1] - self.pre_sum[left]        

if __name__ == '__main__':
    # Your NumArray object will be instantiated and called as such:
    # obj = NumArray(nums)
    # param_1 = obj.sumRange(left,right)
    pass