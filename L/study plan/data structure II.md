### 54. Spiral Matrix

Given an `m x n` `matrix`, return *all elements of the* `matrix` *in spiral order*.

 

**Example 1:**

![img](image_backup/data structure II/spiral1.jpg)

```
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]
```

**Example 2:**

![img](image_backup/data structure II/spiral.jpg)

```
Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]
```

 

**Constraints:**

- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= m, n <= 10`
- `-100 <= matrix[i][j] <= 100`

```python
def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
    dir = [[0, 1], [1, 0], [0, -1], [-1, 0]]

    x, y = 0, -1
    ret = list()
    dir_idx = 0
    m, n = len(matrix), len(matrix[0])

    step = list()
    while m >= 0 and n >= 0:
        step += [n, m - 1]
        m -= 1
        n -= 1
    step_idx = 0

    while True:
        if step[step_idx] == 0: break

        for i in range(step[step_idx]):
            x, y = x + dir[dir_idx][0], y + dir[dir_idx][1]
            ret.append(matrix[x][y])
        dir_idx = (dir_idx + 1) % 4
        step_idx += 1
    return ret
```

### 1630. Arithmetic Subarrays

A sequence of numbers is called **arithmetic** if it consists of at least two elements, and the difference between every two consecutive elements is the same. More formally, a sequence `s` is arithmetic if and only if `s[i+1] - s[i] == s[1] - s[0] `for all valid `i`.

For example, these are **arithmetic** sequences:

```
1, 3, 5, 7, 9
7, 7, 7, 7
3, -1, -5, -9
```

The following sequence is not **arithmetic**:

```
1, 1, 2, 5, 7
```

You are given an array of `n` integers, `nums`, and two arrays of `m` integers each, `l` and `r`, representing the `m` range queries, where the `ith` query is the range `[l[i], r[i]]`. All the arrays are **0-indexed**.

Return *a list of* `boolean` *elements* `answer`*, where* `answer[i]` *is* `true` *if the subarray* `nums[l[i]], nums[l[i]+1], ... , nums[r[i]]` *can be **rearranged** to form an **arithmetic** sequence, and* `false` *otherwise.*

 

**Example 1:**

```
Input: nums = [4,6,5,9,3,7], l = [0,0,2], r = [2,3,5]
Output: [true,false,true]
Explanation:
In the 0th query, the subarray is [4,6,5]. This can be rearranged as [6,5,4], which is an arithmetic sequence.
In the 1st query, the subarray is [4,6,5,9]. This cannot be rearranged as an arithmetic sequence.
In the 2nd query, the subarray is [5,9,3,7]. This can be rearranged as [3,5,7,9], which is an arithmetic sequence.
```

**Example 2:**

```
Input: nums = [-12,-9,-3,-12,-6,15,20,-25,-20,-15,-10], l = [0,1,6,4,8,7], r = [4,4,9,7,9,10]
Output: [false,true,false,false,true,true]
```

 

**Constraints:**

- `n == nums.length`
- `m == l.length`
- `m == r.length`
- `2 <= n <= 500`
- `1 <= m <= 500`
- `0 <= l[i] < r[i] < n`
- `-10 ** 5 <= nums[i] <= 10 ** 5`

```python
def checkArithmeticSubarrays(self, nums: List[int], l: List[int], r: List[int]) -> List[bool]:
    ret = list()
    for start, end in zip(l, r):
        temp = sorted(nums[start : end + 1])
        diff = {j - i for i, j in zip(temp, temp[1:])}
        ret.append(len(diff) == 1)
    return ret
```

### 503. Next Greater Element II

Given a circular integer array `nums` (i.e., the next element of `nums[nums.length - 1]` is `nums[0]`), return *the **next greater number** for every element in* `nums`.

The **next greater number** of a number `x` is the first greater number to its traversing-order next in the array, which means you could search circularly to find its next greater number. If it doesn't exist, return `-1` for this number.

 

**Example 1:**

```
Input: nums = [1,2,1]
Output: [2,-1,2]
Explanation: The first 1's next greater number is 2; 
The number 2 can't find next greater number. 
The second 1's next greater number needs to search circularly, which is also 2.
```

**Example 2:**

```
Input: nums = [1,2,3,4,3]
Output: [2,3,4,-1,4]
```

 

**Constraints:**

- `1 <= nums.length <= 10 ** 4`
- `-10 ** 9 <= nums[i] <= 10 ** 9`

```python
def nextGreaterElements(self, nums: List[int]) -> List[int]:
    n = len(nums)
    ret = [float('inf')] * n
    nums = nums + nums
    pq = []
    for idx, i in enumerate(nums):
        while pq and pq[0][0] < i:
            ret[heapq.heappop(pq)[1]] = i
        if idx < n:
            heapq.heappush(pq, (i, idx))
        else:
            if not pq: break
    return [i if i < float('inf') else -1 for i in ret]
```

