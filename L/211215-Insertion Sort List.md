Given the `head` of a singly linked list, sort the list using **insertion sort**, and return *the sorted list's head*.

The steps of the **insertion sort** algorithm:

1. Insertion sort iterates, consuming one input element each repetition and growing a sorted output list.
2. At each iteration, insertion sort removes one element from the input data, finds the location it belongs within the sorted list and inserts it there.
3. It repeats until no input elements remain.

The following is a graphical example of the insertion sort algorithm. The partially sorted list (black) initially contains only the first element in the list. One element (red) is removed from the input data and inserted in-place into the sorted list with each iteration.

![img](https://upload.wikimedia.org/wikipedia/commons/0/0f/Insertion-sort-example-300px.gif)

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/03/04/sort1linked-list.jpg)

```
Input: head = [4,2,1,3]
Output: [1,2,3,4]
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2021/03/04/sort2linked-list.jpg)

```
Input: head = [-1,5,3,4,0]
Output: [-1,0,3,4,5]
```

 

**Constraints:**

- The number of nodes in the list is in the range `[1, 5000]`.
- `-5000 <= Node.val <= 5000`

(人 •͈ᴗ•͈)

```python
def insertionSortList(self, head: ListNode) -> ListNode:
    pre = ListNode(-1) # pre is a dummy node
    pre.next = to_insert = head
    while head and head.next:
        if head.val > head.next.val:
            # remove head.next and insert it somewhere else
            to_insert = ListNode(head.next.val)
            # point head.next to head.next.next
            head.next = head.next.next
            
            # find the corresponding position to insert
            find_pos = pre
            while find_pos.next.val < to_insert.val: 
                find_pos = find_pos.next
            # insert between find_post and find_pos.next
            temp = find_pos.next
            find_pos.next, to_insert.next = to_insert, temp
        else:
            head = head.next
    return pre.next
```

Runtime: 176 ms, faster than 86.87% of Python3 online submissions for Insertion Sort List.

Memory Usage: 16.4 MB, less than 69.39% of Python3 online submissions for Insertion Sort List.