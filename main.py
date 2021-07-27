#!/usr/bin/env python3
from tricks import largestNumber
from solution import ListNode, TreeNode, Solution

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
    ListNode.printList(solu.reverseKGroup(a, 2))
    p = solu.shuffle([1,2,3,4,5,6])
    print(p.shuffle())
    print(p.reset())
    solu.sortedArrayToBST([1, 2, 3, 4, 5, 7, 10]).printNode()
    print(solu.threeSumClosest([0, -1, -1, 2, 3, 6, -4, -2, 3, 5, -1], -3))
if __name__ == "__main__":
    main()