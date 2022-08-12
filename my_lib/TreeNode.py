# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

    def printNode(self):
        if self:
            print(self.val)
        if self.left:
            self.left.printNode()
        if self.right:
            self.right.printNode()
