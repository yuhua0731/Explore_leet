Given the `head` of a linked list and an integer `val`, remove all the nodes of the linked list that has `Node.val == val`, and return *the new head*.

![img](https://assets.leetcode.com/uploads/2021/03/06/removelinked-list.jpg)

### My first solution: recursion

```python
def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
    def check(curr: ListNode):
        if not curr: return None
        if curr.val == val: return check(curr.next)
        else:
            curr.next = check(curr.next)
            return curr
    return check(head)
```

Runtime: 80 ms, faster than 33.20% of Python3 online submissions for Remove Linked List Elements.

Memory Usage: 28.2 MB, less than 5.21% of Python3 online submissions for Remove Linked List Elements.

### Second approach: two pointers

```python
def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
    pre_ans = pre = ListNode(-1)
    pre.next = curr = head
    while curr:
        if curr.val == val:
            pre.next = curr = curr.next
        else:
            pre, curr = curr, curr.next
    return pre_ans.next
```

Runtime: 60 ms, faster than 97.00% of Python3 online submissions for Remove Linked List Elements.

Memory Usage: 17.3 MB, less than 26.69% of Python3 online submissions for Remove Linked List Elements.