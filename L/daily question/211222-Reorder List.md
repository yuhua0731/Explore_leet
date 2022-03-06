You are given the head of a singly linked-list. The list can be represented as:

```
L0 → L1 → … → Ln - 1 → Ln
```

*Reorder the list to be on the following form:*

```
L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
```

You may not modify the values in the list's nodes. Only nodes themselves may be changed.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/03/04/reorder1linked-list.jpg)

```
Input: head = [1,2,3,4]
Output: [1,4,2,3]
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2021/03/09/reorder2-linked-list.jpg)

```
Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]
```

 

**Constraints:**

- The number of nodes in the list is in the range `[1, 5 * 104]`.
- `1 <= Node.val <= 1000`

```python
def reorderList(self, head: ListNode) -> None:
    """
    Do not return anything, modify head in-place instead.
    """
    n, head_t, tail = 0, head, None
    while head_t:
        n += 1
        temp = ListNode(head_t.val)
        temp.next = tail
        tail = temp
        head_t = head_t.next
    # now we have head & tail, with total amount n
    ans = head
    tail_next, head_next = tail, head
    for _ in range(n >> 1):
        tail = tail_next
        head_next = head.next
        tail_next = tail.next
        head.next = tail
        tail.next = head_next
        head = head_next
    if n & 1: head.next = None
    else: tail.next = None
    return ans
```

passed, but hard to read

#### Discussion

3-step solution

1. Find middle point
2. Reverse second half list
3. Merge two lists

```python
def reorderList(self, head):
    #step 1: find middle
    if not head: return []
    slow, fast = head, head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    #step 2: reverse second half
    prev, curr = None, slow.next
    while curr:
        nextt = curr.next
        curr.next = prev
        prev = curr
        curr = nextt    
    slow.next = None

    #step 3: merge lists
    head1, head2 = head, prev
    while head2:
        nextt = head1.next
        head1.next = head2
        head1 = head2
        head2 = nextt
```

