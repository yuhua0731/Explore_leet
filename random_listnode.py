#!/usr/bin/env python3
from random import randrange

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:

    def __init__(self, head: ListNode):
        self.rand = list()
        while head:
            self.rand += [head.val]
            head = head.next
        self.size = len(self.rand)

    def getRandom(self) -> int:
        return self.rand[randrange(self.size)]

if __name__ == '__main__':
    # Your Solution object will be instantiated and called as such:
    head = ListNode(1)
    head.next = ListNode(2)
    head.next.next = ListNode(3)
    obj = Solution(head)
    print(obj.getRandom())
    print(obj.getRandom())
    print(obj.getRandom())