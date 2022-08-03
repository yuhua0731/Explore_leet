Given the `head` of a singly linked list and two integers `left` and `right` where `left <= right`, reverse the nodes of the list from position `left` to position `right`, and return *the reversed list*.

 

**Example 1:**

![img](image_backup/220721-Reverse Linked List II/rev2ex2.jpg)

```
Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]
```

**Example 2:**

```
Input: head = [5], left = 1, right = 1
Output: [5]
```

 

**Constraints:**

- The number of nodes in the list is `n`.
- `1 <= n <= 500`
- `-500 <= Node.val <= 500`
- `1 <= left <= right <= n`

 

**Follow up:** Could you do it in one pass?

> this ListNode is 1-indexed

```python
def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
    if left == right: return head
    curr = dummy = ListNode(-1)
    dummy.next = head
    for _ in range(m - 1):
        curr = curr.next
    tail = curr.next 
    # at this point, curr is the one before the first reverse node, tail is the first node that will be reversed

    for _ in range(n - m):
        tmp = curr.next
        curr.next = tail.next
        tail.next = tail.next.next
        curr.next.next = tmp
    return dummy.next
```

