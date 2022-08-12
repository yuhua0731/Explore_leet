#!/usr/bin/env python3
from typing import List
import random

class shuffle:
    def __init__(self, nums: List[int]):
        self.origin = nums

    def reset(self) -> List[int]:
        return self.origin

    def shuffle(self) -> List[int]:
        return random.sample(self.origin, len(self.origin))

        
if __name__ == '__main__':
# Your Solution object will be instantiated and called as such:
    obj = shuffle([1, 4, 6, 10, 3, 29])
    print(obj.reset())
    print(obj.shuffle())