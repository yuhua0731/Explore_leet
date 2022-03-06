Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)

 

**Example 1:**

![img](image_backup/220216-Swap Nodes in Pairs/swap_ex1.jpg)

```
Input: head = [1,2,3,4]
Output: [2,1,4,3]
```

**Example 2:**

```
Input: head = []
Output: []
```

**Example 3:**

```
Input: head = [1]
Output: [1]
```

 

**Constraints:**

- The number of nodes in the list is in the range `[0, 100]`.
- `0 <= Node.val <= 100`

```python
def swapPairs(self, head: ListNode) -> ListNode:
    ans = ListNode()
    ans.next = head

    pre, curr = ans, head
    while curr and curr.next:
        nxt, remain = curr.next, curr.next.next

        # swap middle two nodes
        pre.next = nxt
        nxt.next = curr
        curr.next = remain

        # reassign all pointers
        pre = curr
        curr = remain
    return ans.next
```

