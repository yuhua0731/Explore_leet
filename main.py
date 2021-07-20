#!/usr/bin/env python3
from tricks import largestNumber
from solution import ListNode, Solution

def main():
    solu = Solution()
    print(solu.isIsomorphic('apple', 'heelo'))
    print(solu.numDecodings('1234**01*0'))
    print(solu.lengthOfLIS([1, 3, 6, 7, 9, 4, 10, 5, 6]))
    print(solu.findLength([1,2,3,2,1], [3,2,1,4,7]))
    print(solu.findPeakElement([1, 2, 5, 3, 7, 4, 10]))
    print(solu.fourSum([1,0,-1,0,-2,2], 0))
    print(solu.triangleNumber([2,2,3,4]))
    a = ListNode(1, next=ListNode(2, next=ListNode(3, next=ListNode(4, next=ListNode(5)))))
    print(ListNode.printList(solu.reverseKGroup(a, 2)))
if __name__ == "__main__":
    main()