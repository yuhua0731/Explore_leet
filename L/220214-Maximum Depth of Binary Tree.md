Given the `root` of a binary tree, return *its maximum depth*.

A binary tree's **maximum depth** is the number of nodes along the longest path from the root node down to the farthest leaf node.

 

**Example 1:**

![img](image_backup/220214-Maximum Depth of Binary Tree/tmp-tree.jpg)

```
Input: root = [3,9,20,null,null,15,7]
Output: 3
```

**Example 2:**

```
Input: root = [1,null,2]
Output: 2
```

 

**Constraints:**

- The number of nodes in the tree is in the range `[0, 104]`.
- `-100 <= Node.val <= 100`

```python
def maxDepth(self, root: TreeNode) -> int:
    def depth(node):
        return 0 if not node else 1 + max(depth(node.left), depth(node.right))
    return depth(root)
```

