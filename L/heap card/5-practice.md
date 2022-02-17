# practice

## Kth Largest Element in an Array

Given an integer array `nums` and an integer `k`, return *the* `kth` *largest element in the array*.

Note that it is the `kth` largest element in the sorted order, not the `kth` distinct element.

 

**Example 1:**

```
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
```

**Example 2:**

```
Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4
```

 

**Constraints:**

- `1 <= k <= nums.length <= 104`
- `-104 <= nums[i] <= 104`

```python
def findKthLargest(self, nums: List[int], k: int) -> int:
    nums = [-i for i in nums]
    heapq.heapify(nums)
    for _ in range(k):
        ans = heapq.heappop(nums)
    return -ans
```



## Top K Frequent Elements

Given an integer array `nums` and an integer `k`, return *the* `k` *most frequent elements*. You may return the answer in **any order**.

 

**Example 1:**

```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
```

**Example 2:**

```
Input: nums = [1], k = 1
Output: [1]
```

 

**Constraints:**

- `1 <= nums.length <= 10 ** 5`
- `k` is in the range `[1, the number of unique elements in the array]`.
- It is **guaranteed** that the answer is **unique**.

 

**Follow up:** Your algorithm's time complexity must be better than `O(n log n)`, where n is the array's size.

```python
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    nums = [(-cnt, n) for n, cnt in collections.Counter(nums).items()]
    heapq.heapify(nums)
    return [heapq.heappop(nums)[1] for _ in range(k)]
```



## Kth Largest Element in a Stream

Design a class to find the `kth` largest element in a stream. Note that it is the `kth` largest element in the sorted order, not the `kth` distinct element.

Implement `KthLargest` class:

- `KthLargest(int k, int[] nums)` Initializes the object with the integer `k` and the stream of integers `nums`.
- `int add(int val)` Appends the integer `val` to the stream and returns the element representing the `kth` largest element in the stream.

 

**Example 1:**

```
Input
["KthLargest", "add", "add", "add", "add", "add"]
[[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]
Output
[null, 4, 5, 5, 8, 8]

Explanation
KthLargest kthLargest = new KthLargest(3, [4, 5, 8, 2]);
kthLargest.add(3);   // return 4
kthLargest.add(5);   // return 5
kthLargest.add(10);  // return 5
kthLargest.add(9);   // return 8
kthLargest.add(4);   // return 8
```

 

**Constraints:**

- `1 <= k <= 10 ** 4`
- `0 <= nums.length <= 10 ** 4`
- `-10 ** 4 <= nums[i] <= 10 ** 4`
- `-10 ** 4 <= val <= 10 ** 4`
- At most `10 ** 4` calls will be made to `add`.
- It is guaranteed that there will be at least `k` elements in the array when you search for the `kth` element.



> Heapq only guarantee the root node to be the max or min, whreas, it does not care the following elements. Hence, it is not a sorted data type.

```python
class KthLargest:
    """
    a class to find the kth largest element in a list
    min-heap
    """
    def __init__(self, k: int, nums: List[int]):
        self.pq = [] # heapq to store the k largest elements
        nums.append(- 10 ** 4 - 1)
        for i in nums:
            if len(self.pq) == k:
                heapq.heappushpop(self.pq, i)
            else:
                heapq.heappush(self.pq, i)

    def add(self, val: int) -> int:
        heapq.heappushpop(self.pq, val)
        return self.pq[0]
```



## Last Stone Weight

You are given an array of integers `stones` where `stones[i]` is the weight of the `ith` stone.

We are playing a game with the stones. On each turn, we choose the **heaviest two stones** and smash them together. Suppose the heaviest two stones have weights `x` and `y` with `x <= y`. The result of this smash is:

- If `x == y`, both stones are destroyed, and
- If `x != y`, the stone of weight `x` is destroyed, and the stone of weight `y` has new weight `y - x`.

At the end of the game, there is **at most one** stone left.

Return *the smallest possible weight of the left stone*. If there are no stones left, return `0`.

 

**Example 1:**

```
Input: stones = [2,7,4,1,8,1]
Output: 1
Explanation: 
We combine 7 and 8 to get 1 so the array converts to [2,4,1,1,1] then,
we combine 2 and 4 to get 2 so the array converts to [2,1,1,1] then,
we combine 2 and 1 to get 1 so the array converts to [1,1,1] then,
we combine 1 and 1 to get 0 so the array converts to [1] then that's the value of the last stone.
```

**Example 2:**

```
Input: stones = [1]
Output: 1
```

 

**Constraints:**

- `1 <= stones.length <= 30`
- `1 <= stones[i] <= 1000`

> Hide Hint #1 

Simulate the process. We can do it with a heap, or by sorting some list of stones every time we take a turn.

```python
def lastStoneWeight(self, stones: List[int]) -> int:
    pq = [-s for s in stones]
    heapq.heapify(pq)
    while len(pq) > 1:
        y, x = -heapq.heappop(pq), -heapq.heappop(pq)
        if x == y:
            continue
        else:
            heapq.heappush(pq, -(y - x))

    return 0 if not pq else -pq[0]
```



## The K Weakest Rows in a Matrix

You are given an `m x n` binary matrix `mat` of `1`'s (representing soldiers) and `0`'s (representing civilians). The soldiers are positioned **in front** of the civilians. That is, all the `1`'s will appear to the **left** of all the `0`'s in each row.

A row `i` is **weaker** than a row `j` if one of the following is true:

- The number of soldiers in row `i` is less than the number of soldiers in row `j`.
- Both rows have the same number of soldiers and `i < j`.

Return *the indices of the* `k` ***weakest** rows in the matrix ordered from weakest to strongest*.

 

**Example 1:**

```
Input: mat = 
[[1,1,0,0,0],
 [1,1,1,1,0],
 [1,0,0,0,0],
 [1,1,0,0,0],
 [1,1,1,1,1]], 
k = 3
Output: [2,0,3]
Explanation: 
The number of soldiers in each row is: 
- Row 0: 2 
- Row 1: 4 
- Row 2: 1 
- Row 3: 2 
- Row 4: 5 
The rows ordered from weakest to strongest are [2,0,3,1,4].
```

**Example 2:**

```
Input: mat = 
[[1,0,0,0],
 [1,1,1,1],
 [1,0,0,0],
 [1,0,0,0]], 
k = 2
Output: [0,2]
Explanation: 
The number of soldiers in each row is: 
- Row 0: 1 
- Row 1: 4 
- Row 2: 1 
- Row 3: 1 
The rows ordered from weakest to strongest are [0,2,3,1].
```

 

**Constraints:**

- `m == mat.length`
- `n == mat[i].length`
- `2 <= n, m <= 100`
- `1 <= k <= m`
- `matrix[i][j]` is either 0 or 1.

  Hide Hint #1 

Sort the matrix row indexes by the number of soldiers and then row indexes.

> Heapq: element is represented as [-soldier count, -row index]

```python
def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
    # anti-weakest heap
    # element = (-cnt, -index)
    pq = []
    for idx, row in enumerate(mat):
        cnt = sum(row)
        if len(pq) < k:
            heapq.heappush(pq, (-cnt, -idx))
        else:
            heapq.heappushpop(pq, (-cnt, -idx))
    ans = []
    while pq:
        ans.append(-heapq.heappop(pq)[1])
    return ans[::-1]
```



## Kth Smallest Element in a Sorted Matrix

Given an `n x n` `matrix` where each of the rows and columns is sorted in ascending order, return *the* `kth` *smallest element in the matrix*.

Note that it is the `kth` smallest element **in the sorted order**, not the `kth` **distinct** element. 

You must find a solution with a memory complexity better than `O(n2)`.

 

**Example 1:**

```
Input: matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
Output: 13
Explanation: The elements in the matrix are [1,5,9,10,11,12,13,13,15], and the 8th smallest number is 13
```

**Example 2:**

```
Input: matrix = [[-5]], k = 1
Output: -5
```

 

**Constraints:**

- `n == matrix.length == matrix[i].length`
- `1 <= n <= 300`
- `-109 <= matrix[i][j] <= 109`
- All the rows and columns of `matrix` are **guaranteed** to be sorted in **non-decreasing order**.
- `1 <= k <= n2`

 

**Follow up:**

- Could you solve the problem with a constant memory (i.e., `O(1)` memory complexity)?
- Could you solve the problem in `O(n)` time complexity? The solution may be too advanced for an interview but you may find reading [this paper](http://www.cse.yorku.ca/~andy/pubs/X+Y.pdf) fun.

```python
def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
    # heapq: element = (number, row index, col index)
    m, n = len(matrix), len(matrix[0])
    pq = [(matrix[i][0], i, 0) for i in range(m)]
    heapq.heapify(pq)

    for _ in range(k):
        i, row, col = heapq.heappop(pq)
        if col + 1 < n:
            heapq.heappush(pq, (matrix[row][col + 1], row, col + 1))
    return i 
```



## K Closest Points to Origin

Given an array of `points` where `points[i] = [xi, yi]` represents a point on the **X-Y** plane and an integer `k`, return the `k` closest points to the origin `(0, 0)`.

The distance between two points on the **X-Y** plane is the Euclidean distance (i.e., `√(x1 - x2)2 + (y1 - y2)2`).

You may return the answer in **any order**. The answer is **guaranteed** to be **unique** (except for the order that it is in).

 

**Example 1:**

![img](image_backup/5-practice/closestplane1.jpg)

```
Input: points = [[1,3],[-2,2]], k = 1
Output: [[-2,2]]
Explanation:
The distance between (1, 3) and the origin is sqrt(10).
The distance between (-2, 2) and the origin is sqrt(8).
Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
We only want the closest k = 1 points from the origin, so the answer is just [[-2,2]].
```

**Example 2:**

```
Input: points = [[3,3],[5,-1],[-2,4]], k = 2
Output: [[3,3],[-2,4]]
Explanation: The answer [[-2,4],[3,3]] would also be accepted.
```

 

**Constraints:**

- `1 <= k <= points.length <= 10 ** 4`
- `-10 ** 4 < xi, yi < 10 ** 4`

```python
def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
    # anti-closest heapq: element = (distance, x, y)
    pq = []
    for x, y in points:
        if len(pq) < k:
            heapq.heappush(pq, (-(x ** 2 + y ** 2) ** 0.5, x, y))
        else:
            heapq.heappushpop(pq, (-(x ** 2 + y ** 2) ** 0.5, x, y))
    return [[x, y] for _, x, y in pq]
```



## Furthest Building You Can Reach

You are given an integer array `heights` representing the heights of buildings, some `bricks`, and some `ladders`.

You start your journey from building `0` and move to the next building by possibly using bricks or ladders.

While moving from building `i` to building `i+1` (**0-indexed**),

- If the current building's height is **greater than or equal** to the next building's height, you do **not** need a ladder or bricks.
- If the current building's height is **less than** the next building's height, you can either use **one ladder** or `(h[i+1] - h[i])` **bricks**.

*Return the furthest building index (0-indexed) you can reach if you use the given ladders and bricks optimally.*

 

**Example 1:**

![img](image_backup/5-practice/q4.gif)

```
Input: heights = [4,2,7,6,9,14,12], bricks = 5, ladders = 1
Output: 4
Explanation: Starting at building 0, you can follow these steps:
- Go to building 1 without using ladders nor bricks since 4 >= 2.
- Go to building 2 using 5 bricks. You must use either bricks or ladders because 2 < 7.
- Go to building 3 without using ladders nor bricks since 7 >= 6.
- Go to building 4 using your only ladder. You must use either bricks or ladders because 6 < 9.
It is impossible to go beyond building 4 because you do not have any more bricks or ladders.
```

**Example 2:**

```
Input: heights = [4,12,2,7,3,18,20,3,19], bricks = 10, ladders = 2
Output: 7
```

**Example 3:**

```
Input: heights = [14,3,19,3], bricks = 17, ladders = 0
Output: 3
```

 

**Constraints:**

- `1 <= heights.length <= 105`
- `1 <= heights[i] <= 106`
- `0 <= bricks <= 109`
- `0 <= ladders <= heights.length`

  Hide Hint #1 

Assume the problem is to check whether you can reach the last building or not.

  Hide Hint #2 

You'll have to do a set of jumps, and choose for each one whether to do it using a ladder or bricks. It's always optimal to use ladders in the largest jumps.

  Hide Hint #3 

Iterate on the buildings, maintaining the largest r jumps and the sum of the remaining ones so far, and stop whenever this sum exceeds b.

```python
def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
    # spend ladders for 'ladders' highest jumps
    # sum up the rest jumps, if the sum is less than or equal to bricks, then we can reach here
    jumps = []
    total_bri = 0
    n = len(heights)
    for i in range(n - 1):
        diff = heights[i + 1] - heights[i]
        if diff <= 0: continue
        if len(jumps) < ladders:
            heapq.heappush(jumps, diff)
        else:
            total_bri += heapq.heappushpop(jumps, diff)
            if total_bri > bricks: return i
    return n - 1
```



## Find Median from Data Stream

The **median** is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values.

- For example, for `arr = [2,3,4]`, the median is `3`.
- For example, for `arr = [2,3]`, the median is `(2 + 3) / 2 = 2.5`.

Implement the MedianFinder class:

- `MedianFinder()` initializes the `MedianFinder` object.
- `void addNum(int num)` adds the integer `num` from the data stream to the data structure.
- `double findMedian()` returns the median of all elements so far. Answers within `10-5` of the actual answer will be accepted.

 

**Example 1:**

```
Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output
[null, null, null, 1.5, null, 2.0]

Explanation
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
```

 

**Constraints:**

- `-10 ** 5 <= num <= 10 ** 5`
- There will be at least one element in the data structure before calling `findMedian`.
- At most `5 * 10 ** 4` calls will be made to `addNum` and `findMedian`.

 

**Follow up:**

- If all integer numbers from the stream are in the range `[0, 100]`, how would you optimize your solution?
- If `99%` of all integer numbers from the stream are in the range `[0, 100]`, how would you optimize your solution?

```python
import heapq

class MedianFinder:

    def __init__(self):
        self.size = 0
        self.upper_pq = [] # min heap
        self.lower_pq = [] # max heap

    def addNum(self, num: int) -> None:
        self.size += 1
        if self.size % 2:
            # total size is odd
            # get max element from lower, then push to upper
            heapq.heappush(self.upper_pq, -heapq.heappushpop(self.lower_pq, -num))
        else:
            # total size is even
            # get min element from upper, then push to lower
            heapq.heappush(self.lower_pq, -heapq.heappushpop(self.upper_pq, num))

    def findMedian(self) -> float:
        if self.size % 2:
            return self.upper_pq[0]
        else:
            return (self.upper_pq[0] + self.lower_pq[0]) / 2

if __name__ == '__main__':
    # Your MedianFinder object will be instantiated and called as such:
    obj = MedianFinder()
    obj.addNum(-1)
    print(obj.findMedian())
    obj.addNum(-2)
    print(obj.findMedian())
    obj.addNum(-3)
    print(obj.findMedian())
    obj.addNum(-4)
    print(obj.findMedian())
    obj.addNum(-5)
    print(obj.findMedian())
```

