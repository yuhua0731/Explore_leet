# Explore_leet
## 1. heapq in python
heap queue, aka priority queue, always keep all elements sorted with smallest element in the first position
### 2 different properties:
- zero-based indexing, h[i]'s children are h[2 * i + 1] and h[2 * i + 2]
- min-heap, pop method returns the smallest element
tip: if you want to implement a max heap, just negative all elements
### common use functions
- initialize: h = [] or h = heapify(list)
- heappush(heap, item)
- heappop(heap)
- heappushpop(heap, item): push item first, then pop the smallest item
- heapreplace(heap, item): pop smallest item first, then push item
- merge(list1, list2)
- merge(*iterables, key=None, reverse=False) do not know how to use...
- nlargest/nsmallest: return a list

## 2. pointer operations in python
this is really confusing to me at the first time, but after thinking about what we do in programming C, all things become extremely clear to me.
### ListNode
```Python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```
ListNode in python is worldwidely used as LinkedList in Java
Now, think about the following codes:
```Python
cur = ListNode(1, next=ListNode(2, next=ListNode(3)))
# cur: 1 -> 2 -> 3 -> None
head = jump = ListNode(0)
# head = jump: 0 -> None

# head is the real leader, never got changed, a pointer which point to the first node
# head.next is where we store our result

head.next = l = cur # pre.next = cur as well
#       0     ->    1    ->    2    ->    3   ->   None
#       ↑           ↑                     
#  head & jump      l                     

# after several operations, imagine a situation like this:
#           r ->    3    ->    None
#                   ↑
#       0     ->    1    <-    2 
#       ↑           ↑          ↑ 
#  head & jump      l         pre

jump.next = pre
#       0     ->    2    ->    1    ->    3   ->   None
#       ↑           ↑          ↑          ↑
#  head & jump     pre         l          r

jump = l 
# this is confusing to me at first
# this statement does not replace jump with l
# it changes jump pointer, pointing to where l points
# hence, head will not be affected
#       0     ->    2    ->    1    ->    3   ->   None
#       ↑           ↑          ↑          ↑
#     head         pre       l & jump     r

l = r
# this is the same as above one, just change pointer l's location
#       0     ->    2    ->    1    ->    3   ->   None
#       ↑           ↑          ↑          ↑
#     head         pre        jump      l & r
```

## 3. Binary Search Tree, BTS for short
also called an **ordered**, or **sorted** search tree.

node.val is greater than all values from its left subtree, and less than all values from its right subtree