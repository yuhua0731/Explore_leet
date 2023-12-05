# deque

## python collections.deque

```python
from collections import deque

deque()
deque('abc')
deque(['a', 'b', 'c'])

llist = deque('abc')
llist.append('f')
llist.appendleft('f')
llist.pop()
llist.popleft()
```

## Self implemented LinkedList

> 阅读材料：https://realpython.com/linked-lists-python/

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None
    
    def __str__(self):
        return str(self.val)
    
    def __repr__(self):
        return str(self.val)

class LinkedList:
    def __init__(self, nodes=None):
        self.size = len(nodes) if nodes else 0
        if nodes is not None:
            node = Node(val=nodes.pop(0))
            self.head = node
            for elem in nodes:
                node.next = Node(val=elem)
                node = node.next
            self.tail = node
        else:
            self.head = None
            self.tail = None

    def __repr__(self) -> str:
        node = self.head
        nodes = []
        while node:
            nodes.append(str(node))
            node = node.next
        return '->'.join(nodes)
    
    def __iter__(self):
        node = self.head
        while node is not None:
            yield node
            node = node.next

    def addFirst(self, val):
        node = Node(val)
        if self.size == 0:
            self.head = node
            self.tail = node
        else:
            node.next = self.head
            self.head = node
        self.size += 1

    def addLast(self, val):
        node = Node(val)
        if self.size == 0:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = self.tail.next
        self.size += 1

    def removeFirst(self):
        if self.size == 0:
            raise Exception("LinkedList is empty")
        if self.size == 1:
            self.head = None
            self.tail = None
        else:
            self.head = self.head.next
        self.size -= 1

    def removeLast(self):
        if self.size == 0:
            raise Exception("LinkedList is empty")
        if self.size == 1:
            self.head = None
            self.tail = None
        else:
            prev_node = self.head
            while prev_node.next != self.tail:
                prev_node = prev_node.next
            prev_node.next = None
            self.tail = prev_node
        self.size -= 1

    def addAfter(self, target_node_data, new_node):
        if not self.head:
            raise Exception("LinkedList is empty")

        for node in self:
            if node.val == target_node_data:
                new_node.next = node.next
                node.next = new_node
                self.size += 1
                return

        raise Exception("Node with data '%s' not found" % target_node_data)
    
    def addBefore(self, target_node_data, new_node):
        if self.head is None:
            raise Exception("List is empty")

        if self.head.val == target_node_data:
            return self.addFirst(new_node)

        prev_node = self.head
        while prev_node.next:
            if prev_node.next.val == target_node_data:
                new_node.next = prev_node.next
                prev_node.next = new_node
                self.size += 1
                return
            prev_node = prev_node.next

        raise Exception("Node with data '%s' not found" % target_node_data)
    
    def removeNode(self, target_node_data):
        if self.head is None:
            raise Exception("List is empty")
        
        if self.size == 1:
            if self.head.val == target_node_data:
                self.head = None
                self.tail = None
                self.size -= 1
                return
        elif self.head.val == target_node_data:
            self.removeFirst()
        elif self.tail.val == target_node_data:
            self.removeLast()
        else:        
            prev_node = self.head
            while prev_node.next:
                if prev_node.next.val == target_node_data:
                    prev_node.next = prev_node.next.next
                    self.size -= 1
                    return
                prev_node = prev_node.next

# example
ll = LinkedList([100, 101])
ll.addFirst(1)
ll.addFirst(2)
ll.addLast(3)
print(ll, ll.size)  # 2->1->100->101->3 5
for n in ll: print(n) # implement __iter__ will let this work
ll.addAfter(100, Node(102))
print(ll, ll.size)
ll.addBefore(100, Node(99))
print(ll, ll.size)
ll.removeNode(99)
print(ll, ll.size)
```

