#!/usr/bin/env python3
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, start, end):
        self.s = start
        self.e = end
        self.left = None
        self.right = None

class MyCalendar:

    def __init__(self):
        self.event = None

    def book(self, start: int, end: int) -> bool:
        def book_helper(node: TreeNode):
            if node.s >= end:
                if node.left: return book_helper(node.left)
                else: 
                    node.left = TreeNode(start, end)
                    return True
            elif node.e <= start:
                if node.right: return book_helper(node.right)
                else:
                    node.right = TreeNode(start, end)
                    return True
            else:
                return False
        if not self.event: 
            self.event = TreeNode(start, end)
            return True
        else: return book_helper(self.event)

if __name__ == '__main__':
# Your MyCalendar object will be instantiated and called as such:
    obj = MyCalendar()
    input = [[47,50],[33,41],[39,45],[33,42],[25,32],[26,35],[19,25],[3,8],[8,13],[18,27]]
    print([obj.book(start, end) for start, end in input])