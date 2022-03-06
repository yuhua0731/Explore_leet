Given the `root` of a binary tree, find the maximum value `v` for which there exist **different** nodes `a` and `b` where `v = |a.val - b.val|` and `a` is an ancestor of `b`.

A node `a` is an ancestor of `b` if either: any child of `a` is equal to `b` or any child of `a` is an ancestor of `b`.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/11/09/tmp-tree.jpg)

```
Input: root = [8,3,10,1,6,null,14,null,null,4,7,13]
Output: 7
Explanation: We have various ancestor-node differences, some of which are given below :
|8 - 3| = 5
|3 - 7| = 4
|8 - 1| = 7
|10 - 13| = 3
Among all possible differences, the maximum value of 7 is obtained by |8 - 1| = 7.
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2020/11/09/tmp-tree-1.jpg)

```
Input: root = [1,null,2,null,0,3]
Output: 3
```

 

**Constraints:**

- The number of nodes in the tree is in the range `[2, 5000]`.
- `0 <= Node.val <= 105`

#### My approach

```python
def maxAncestorDiff(self, root: TreeNode) -> int:
    def find_diff(node: TreeNode, min_anc: int, max_anc: int):
        if not node: return -1
        diff = [max(abs(node.val - min_anc), abs(node.val - max_anc))]
        min_anc = min(node.val, min_anc)
        max_anc = max(node.val, max_anc)
        diff.append(find_diff(node.left, min_anc, max_anc))
        diff.append(find_diff(node.right, min_anc, max_anc))
        return max(diff)

    return max(find_diff(root.left, root.val, root.val), find_diff(root.right, root.val, root.val))

```

bad performance