# Explore_leet

## heapq in python

heap queue, aka priority queue, always keep all elements sorted with smallest element in the first position

#### two different properties:

- zero-based indexing, `h[i]`'s children are `h[2 * i + 1]` and `h[2 * i + 2]`
- min-heap, pop method returns the smallest element tip: if you want to implement a max heap, just negative all elements

#### common use functions

- initialize: `h = []` or `h = heapify(list)`
- `heappush(heap, item)`
- `heappop(heap)`
- `heappushpop(heap, item)`: push item first, then pop the smallest item
- `heapreplace(heap, item)`: pop smallest item first, then push item
- `merge(list1, list2)`
- `merge(*iterables, key=None, reverse=False)` do not know how to use...
- nlargest/nsmallest: return a list

## pointer operations in python

this is really confusing to me at the first time, but after thinking about what we do in programming C, all things become extremely clear to me.

#### ListNode

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

ListNode in python is worldwidely used as LinkedList in Java Now, think about the following codes:

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

## Binary Search Tree, BTS for short

also called an **ordered**, or **sorted** search tree.

node.val is greater than all values from its left subtree, and less than all values from its right subtree.

## Structural Pattern Matching(PEP 634)

new feature in 3.10 pre.

- no break

```python
match subject:
    case <pattern_1>:
        <action_1>
    case <pattern_2>:
        <action_2>
    case <pattern_3>:
        <action_3>
    case _:
        <action_wildcard>
```

#### Guard

We can add an if clause to a pattern, known as a “guard”. If the guard is false, match goes on to try the next case block. Note that value capture happens before the guard is evaluated:

```python
match point:
    case Point(x, y) if x == y:
        print(f"The point is located on the diagonal Y=X at {x}.")
    case Point(x, y):
        print(f"Point is not on the diagonal.")
```

## List

to use `List` type:

```python
from typing import List
a = ['a', '', '']
a.remove('') # remove one matched element at one time
```

#### remove all empty string from a list

```python
# using list.remove()
while '' in a:
    a.remove('')

# form a new list
a = [i for i in a if i]

# using filter
a = list(filter(None, a))
```

#### filter(function, sequence)

- function: function that tests if each element of a sequence true or not.
- sequence: sequence which needs to be filtered, it can be sets, lists, tuples, or containers of any iterators.
- returns: returns an iterator that is already filtered.

#### append a list to another list

```python
list1.extend(list2)
```

#### merge lists without duplicates

when you want append some elements to a list without causing duplicate items, try this

```python
ans = {1, 2} 
ans |= {2, 3} # ans = {1, 2, 3}
```

## str

- reverse a string
```python
a[::-1]
```

## default dict

```python
d = collections.defaultdict(list)
d = collections.defaultdict(int)
```

## Union-find algorithm

#### 947. Most Stones Removed with Same Row or Column

On a 2D plane, we place n stones at some integer coordinate points. Each coordinate point may have at most one stone.

A stone can be removed if it shares either the same row or the same column as another stone that has not been removed.

Given an array stones of length n where `stones[i] = [xi, yi]` represents the location of the ith stone, return the largest possible number of stones that can be removed.

```java
// union-find algorithm
// each stone is an edge, connect two nodes(one row and one column)
// we can remove all stones except for last one in the same tree(island)
// hence, the left stones equal to the number of trees(islands)
HashMap<Integer, Integer> root = new HashMap<>();
int island = 0;

public int removeStones(int[][] stones) {
    // attention: we need to find a way to distinguish rows and columns
    // since they both start from 1, 2, 3...
    // attempt 1: 1 ~ (N - 1) represent for rows, while N ~ (N + M - 1) for columns
    // attempt 2: 1 ~ (N - 1) for rows, while ~(1 ~ (M - 1)) for columns
    for(int i = 0; i < stones.length; i++)
        union(stones[i][0],  ~stones[i][1]);
    return stones.length - island;
}
int find(int a) {
    // a new island found with single stone(a)
    if(root.putIfAbsent(a, a) == null) island++; 
    if(root.get(a) != a) root.put(a, find(root.get(a)));
    return root.get(a);
}
void union(int a, int b) {
    if(find(a) == find(b)) return;
    root.put(find(a), find(b));
    island--; // two islands got connected
}
```

python version:

```python
# union find
group = [i for i in range(n)] # group[i] = i after initialization

def find(x):
    if group[x] == x: return x
    """
    this line is fucking essential!!! 
    improved time complexity a lot. 
    when you try to find root of x, it takes only 1 step to reach the end. ↓↓↓
    """ 
    group[x] = find(group[x])
    """
    imagine this condition:
    group[2] = 5 -> group[5] = 3 -> group[3] = 7 -> group[7] = 7
    it recursively call find method for four times when you try to find root of 2, everytime, repeatedly.
    what optimization can we do? 
    Actually, after first traversal on 2, we already know that its root and all mid nodes' root are 7. We can direct update group[2] = group[5] = group[3] = group[7] = 7 by adding this line.
    """
    return group[x]

def union(x, y):
    x_group, y_group = find(x), find(y)
    if x_group != y_group:
        group[x_group] = y_group

# pesudo code
for i, j in edge:
    if find[i] == find[j]: continue
    union(i, j)

# now you successfully generate a island map
```

#### 952. Largest Component Size by Common Factor

```python
# union find
n = max(nums)
count = Counter()
group = [i for i in range(n + 1)]

def find(x):
    if group[x] == x: return x
    group[x] = find(group[x])
    return group[x]

def union(x, y):
    x_group, y_group = find(x), find(y)
    if x_group != y_group:
        group[x_group] = y_group

for a in nums:
    for num in range(2, int(math.sqrt(a) + 1)):
        if a % num == 0:
            union(a, num)
            union(a, a // num)

for a in nums: count[find(a)] += 1
return max(count.values())
```

## collections

#### Counter

A Counter is a dict subclass for counting hashable objects. It is a collection where elements are stored as dictionary keys and their counts are stored as dictionary values. Counts are allowed to be any integer value including zero or negative counts.

```python
arr = [1, 1, 7]
count = collections.Counter(arr)
# Counter({1: 2, 7: 1})
```

## sort with customize functions

```python
arr.sort(key=functools.cmp_to_key(lambda x, y: 1 if abs(x) > abs(y) else -1))
arr.sort(key=lambda x: (-x[0], x[1]))
arr.sort(key=abs) # sort original list
sorted(arr, key=abs) # do not affect original list
```

## conversion among diverse object types

```python
i = 10000
b = b'\xc2\xda\x7a\x77'
f = 1.314

# bytes to int
int.from_bytes(b, byteorder = 'little', signed=True)

# int to bytes
i.to_bytes(4, byteorder='little', signed=True)

# bytes to float
# native byte order (little-endian on my machine)
struct.unpack('f', b)[0] # 1.7230105268977664e+16
# big-endian 
struct.unpack('>f', b)[0] # -109.22724914550781

# decode bytes to string:
b.decode("utf-8")

# bytes to hex string
b.hex()

# print 2 decimal places float:
"{:.2f}".format(f)
```

## multiple for loops and if in the same statement

The best way to remember this is that the order of for loop inside the list comprehension is based on the order in which **they appear in traditional loop approach**. Outer most loop comes first, and then the inner loops subsequently.

So, the equivalent list comprehension would be:

```python
[entry for tag in tags for entry in entries if tag in entry]
```

In general, **if-else statement comes before the first for loop**, and if you have just an if statement, it will come at the **end**. For e.g, if you would like to add an empty list, if tag is not in entry, you would do it like this:

```python
[entry if tag in entry else [] for tag in tags for entry in entries]
```

## zip() in python

```python
a =     [0, 1, 2]
a[1:] = [1, 2]
b =     [1, 2, 0]
# here is a little trick: to form a list of (a[i], a[i + 1], b[i])
tuple(zip(a, a[1:], b)) = ((0, 1, 1), (1, 2, 2))
```

## tuple in dict

- tuple is immutable
- tuple can be key as well as value in dict

there are two common ways to modify a tuple:

- generate a new tuple with new value, and replace the elder one
- use += operator:

```python
>>> t = {'k': (1, 2)}
>>> t['k'] += (3,)
>>> t
{'k': (1, 2, 3)}
```

## periodical function

```python
# this function will be called every 2 seconds
def periodical_func():
    print(time.time())
    time.sleep(1)
    threading.Timer(1, periodical_func).start()

periodical_func() # first call here
```

## traverse dict by sorted value

```python
for k, v in sorted(x.items(), key=lambda item: item[1]):
    # your code
```

## manage set

- `.remove()` raise an keyError if element does not exist
- `.discard()` will not raise error
- `.pop()` pop out an arbitrary element, and will raise an error if set is empty
- `.union()` returns a new set with all items from both sets

```python
set3 = set1.union(set2)
```

- `.update()` inserts the items in set2 into set1

```python
set1.update(set2)
```

- Pass

## itertools.chain

It is a function that takes a series of iterables and returns one iterable. It groups all the iterables together and produces a single iterable as output. Its output cannot be used directly and thus explicitly converted into iterables.

```python
# inner implementation
def chain(*iterables):
	for it in iterables:
       	for each in it:
           	yield each

chain (*iterables)

# example
# a list of odd numbers
odd = [1, 3, 5, 7, 9]
# a list of even numbers
even = [2, 4, 6, 8, 10]
# chaining odd and even numbers
numbers = list(chain(odd, even))
# [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
```

just concatenate multiple iterators together without any additional operations.

## itertools.product

In the terms of Mathematics Cartesian Product of two sets is defined as the set of all ordered pairs (a, b) where a belongs to A and b belongs to B.

```python
arr1 = [1, 2, 3] 
arr2 = [5, 6, 7] 
# product(arr1, arr2)
[(1, 5), (1, 6), (1, 7), (2, 5), (2, 6), (2, 7), (3, 5), (3, 6), (3, 7)]

arr1 = [10, 12] 
arr2 = [8, 9, 10] 
# product(arr1, arr2)
[(10, 8), (10, 9), (10, 10), (12, 8), (12, 9), (12, 10)] 
```

- **`itertools.product(*iterables, repeat=1)`:**
  It returns the cartesian product of the provided iterable with itself for the number of times specified by the optional keyword “repeat”. For example, `product(arr, repeat=3)` means the same as `product(arr, arr, arr)`.
- **`itertools.product(*iterables)`:**
  It returns the cartesian product of all the iterable provided as the argument. For example, `product(arr1, arr2, arr3)`.

## String literals

- f’’ represented formatted string
- r’’ represented raw string, which treat backslashes as literal characters
- b’’ byte string, which produce an instance of the [`bytes`](https://docs.python.org/3/library/stdtypes.html#bytes) type instead of the [`str`](https://docs.python.org/3/library/stdtypes.html#str) type

```python
a = 1
print(f'a = {a}')
print(r'/User/huayu/Downloads/')
print(b'\x00\xff')
```

## Jupyter

coding notebook, can view realtime variable states

⇧ ⌘ P: create new blank notebook

![image-20211117204543383](/Users/huayu/Library/Application Support/typora-user-images/image-20211117204543383.png)

## pytest

1. mkdir test && cd test
2. create empty __init\_\_.py file
3. create test_sample.py file and write test cases in this file
4. **==Prepare test cases first, then write your real test function==**

```python
def test_function():
    input_data = '***'
    result = function_to_be_test(input_data)
    assert variable == result
```

## kwarg

[https://book.pythontips.com/en/latest/args_and_kwargs.html](https://book.pythontips.com/en/latest/args_and_kwargs.html)

\*args and \*\*kwargs are mostly used in function definitions. *args and \*\*kwargs allow you to pass an ==unspecified number of arguments== to a function, so when writing the function definition, you do not need to know how many arguments will be passed to your function. \*args is used to send a ==non-keyworded== variable length argument list to the function.

```python
def test_var_args(f_arg, *argv):
    print(f_arg)
    print(argv)
test_var_args('yasoob', 'python', 'eggs', 'test')

"""output
yasoob
('python', 'eggs', 'test')
"""

def greet_me(**kwargs):
    for k, v in kwargs.items():
        print(k, v)
greet_me(a = 1, b = 2)

"""output
a 1
b 2
"""
```

## count non-zero bit

Python 3.10 introduce a new function:[`int.bit_count()`](https://docs.python.org/3.10/library/stdtypes.html#int.bit_count)

This is functionally equivalent to `bin(n).count("1")` but should be [~6 times faster](https://github.com/python/cpython/pull/771#issuecomment-288570744).

## simple command-line http server

```shell
python3 -m http.server 7777
```

## pdf export

选取图像，从pdf中截取矢量图，可以导入画图软件
