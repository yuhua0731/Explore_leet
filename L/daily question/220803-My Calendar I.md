You are implementing a program to use as your calendar. We can add a new event if adding the event will not cause a **double booking**.

A **double booking** happens when two events have some non-empty intersection (i.e., some moment is common to both events.).

The event can be represented as a pair of integers `start` and `end` that represents a booking on the half-open interval `[start, end)`, the range of real numbers `x` such that `start <= x < end`.

Implement the `MyCalendar` class:

- `MyCalendar()` Initializes the calendar object.
- `boolean book(int start, int end)` Returns `true` if the event can be added to the calendar successfully without causing a **double booking**. Otherwise, return `false` and do not add the event to the calendar.

 

**Example 1:**

```
Input
["MyCalendar", "book", "book", "book"]
[[], [10, 20], [15, 25], [20, 30]]
Output
[null, true, false, true]

Explanation
MyCalendar myCalendar = new MyCalendar();
myCalendar.book(10, 20); // return True
myCalendar.book(15, 25); // return False, It can not be booked because time 15 is already booked by another event.
myCalendar.book(20, 30); // return True, The event can be booked, as the first event takes every time less than 20, but not including 20.
```

 

**Constraints:**

- `0 <= start < end <= 109`
- At most `1000` calls will be made to `book`.



#### Straight forward solution

O(N^2^) time complexity

```python
class MyCalendar:

    def __init__(self):
        self.event = list()

    def book(self, start: int, end: int) -> bool:
        for s, e in self.event:
            if s >= end or e <= start: continue
            else: return False
        self.event.append([start, end])
        return True


# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)
```



#### Binary tree

O(logN) time complexity

```python
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
```

