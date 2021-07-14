#!/usr/bin/env python3
from tricks import largestNumber
from solution import Solution

def main():
    solu = Solution()
    print(solu.isIsomorphic('apple', 'heelo'))
    print(solu.numDecodings('1234**01*0'))
    print(solu.lengthOfLIS([1, 3, 6, 7, 9, 4, 10, 5, 6]))
    print(solu.findLength([1,2,3,2,1], [3,2,1,4,7]))
    print(solu.findPeakElement([1, 2, 5, 3, 7, 4, 10]))
if __name__ == "__main__":
    main()