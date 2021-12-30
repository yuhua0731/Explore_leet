### 34. Find First and Last Position of Element in Sorted Array

Given an array of integers `nums` ==sorted== in ==non-decreasing== order, find the starting and ending position of a given `target` value.

If `target` is not found in the array, return `[-1, -1]`.

You must write an algorithm with `O(log n)` runtime complexity.

 

**Example 1:**

```
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
```

**Example 2:**

```
Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
```

**Example 3:**

```
Input: nums = [], target = 0
Output: [-1,-1]
```

 

**Constraints:**

- `0 <= nums.length <= 105`
- `-109 <= nums[i] <= 109`
- `nums` is a non-decreasing array.
- `-109 <= target <= 109`

#### First approach:

Binary search, similar to [Search Insert Position](/Users/huayu/hcrobot/yuhua_test/explore_leet/L/211126-Search Insert Position.md).

The different point is we are going to find the first and last index of target. **We do not break loop when we find an element == target, instead, we stop until left == right**, which means we reached the boarder of target element.

Furthermore, if there is no target element in input list, -1 should be returned.

Implement a `find` function, which can find first and last hit position according to input args.

```python
def searchRange(self, nums: List[int], target: int) -> List[int]:
    def find(first: bool):
        left, right, visited = 0, len(nums), False
        while left < right:
            mid = (left + right) >> 1
            if nums[mid] == target: 
                visited = True
                if first: right = mid
                else: left = mid + 1
            elif nums[mid] > target: right = mid
            elif nums[mid] < target: left = mid + 1
        if not visited: return -1
        return left if first else left - 1
    return [find(True), find(False)]
```

Runtime: 80 ms, faster than 91.21% of Python3 online submissions for Find First and Last Position of Element in Sorted Array.

Memory Usage: 15.3 MB, less than 96.97% of Python3 online submissions for Find First and Last Position of Element in Sorted Array.

### 33. Search in Rotated Sorted Array

There is an integer array `nums` sorted in ascending order (with **distinct** values).

Prior to being passed to your function, `nums` is **possibly rotated** at an unknown pivot index `k` (`1 <= k < nums.length`) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` (**0-indexed**). For example, `[0,1,2,4,5,6,7]` might be rotated at pivot index `3` and become `[4,5,6,7,0,1,2]`.

Given the array `nums` **after** the possible rotation and an integer `target`, return *the index of* `target` *if it is in* `nums`*, or* `-1` *if it is not in* `nums`.

You must write an algorithm with `O(log n)` runtime complexity.

 

**Example 1:**

```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```

**Example 2:**

```
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
```

**Example 3:**

```
Input: nums = [1], target = 0
Output: -1
```

 

**Constraints:**

- `1 <= nums.length <= 5000`
- `-104 <= nums[i] <= 104`
- All values of `nums` are **unique**.
- `nums` is an ascending array that is possibly rotated.
- `-104 <= target <= 104`

#### First approach:

```python
def search(self, nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) >> 1
        if nums[left] == target: return left
        if nums[mid] == target: return mid
        if nums[right] == target: return right
        if nums[mid] > target:
            if target < nums[left] < nums[mid]: left = mid + 1
            else: right = mid
        else:
            if nums[mid] < nums[right] < target: right = mid
            else: left = mid + 1
    return left if nums[left] == target else -1
```

Runtime: 40 ms, faster than 78.86% of Python3 online submissions for Search in Rotated Sorted Array.

Memory Usage: 14.8 MB, less than 24.09% of Python3 online submissions for Search in Rotated Sorted Array.

#### Discussion:

**Explanation**

Let's say `nums` looks like this: [12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

Because it's not fully sorted, we can't do normal binary search. But here comes the trick:

- If target is let's say 14, then we adjust `nums` to this, where "inf" means infinity:
  [12, 13, 14, 15, 16, 17, 18, 19, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf]
- If target is let's say 7, then we adjust `nums` to this:
  [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

And then we can simply do ordinary binary search.

Of course we don't actually adjust the whole array but instead adjust only on the fly only the elements we look at. And the adjustment is done by comparing both the target and the actual element against `nums[0]`.

------

**Code**

If `nums[mid]` and `target` are *"on the same side"* of `nums[0]`, we just take `nums[mid]`. Otherwise we use -infinity or +infinity as needed.

```python
def search(self, nums: List[int], target: int) -> int:
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) >> 1
        temp = nums[mid]
        if (nums[mid] < nums[0]) != (target < nums[0]): 
			# not on the same side
            temp = float('inf') if target > nums[0] else -float('inf')
        if temp == target: return mid
        if temp > target: right = mid
        else: left = mid + 1
    return left
```

Runtime: 44 ms, faster than 55.10% of Python3 online submissions for Search in Rotated Sorted Array.

Memory Usage: 14.6 MB, less than 81.62% of Python3 online submissions for Search in Rotated Sorted Array.

### 74. Search a 2D Matrix

Write an efficient algorithm that searches for a value in an `m x n` matrix. This matrix has the following properties:

- Integers in each row are sorted from left to right.
- The first integer of each row is greater than the last integer of the previous row.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/10/05/mat.jpg)

```
Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2020/10/05/mat2.jpg)

```
Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
Output: false
```

 

**Constraints:**

- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= m, n <= 100`
- `-104 <= matrix[i][j], target <= 104`

```python
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    # time complexity: O(m + log(n)) | space complexity: O(1)
    for row in matrix:
        if row[0] > target: return False
        if row[-1] < target: continue
        left, right = 0, len(row)
        while left < right:
            mid = (left + right) >> 1
            if row[mid] == target: return True
            if row[mid] > target: right = mid
            else: left = mid + 1
    return False
```

Runtime: 44 ms, faster than 69.78% of Python3 online submissions for Search a 2D Matrix.

Memory Usage: 14.6 MB, less than 96.52% of Python3 online submissions for Search a 2D Matrix.

### 153. Find Minimum in Rotated Sorted Array

Suppose an array of length `n` sorted in ascending order is **rotated** between `1` and `n` times. For example, the array `nums = [0,1,2,4,5,6,7]` might become:

- `[4,5,6,7,0,1,2]` if it was rotated `4` times.
- `[0,1,2,4,5,6,7]` if it was rotated `7` times.

Notice that **rotating** an array `[a[0], a[1], a[2], ..., a[n-1]]` 1 time results in the array `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]`.

Given the sorted rotated array `nums` of **unique** elements, return *the minimum element of this array*.

You must write an algorithm that runs in `O(log n) time.`

 

**Example 1:**

```
Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.
```

**Example 2:**

```
Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.
```

**Example 3:**

```
Input: nums = [11,13,15,17]
Output: 11
Explanation: The original array was [11,13,15,17] and it was rotated 4 times. 
```

 

**Constraints:**

- `n == nums.length`
- `1 <= n <= 5000`
- `-5000 <= nums[i] <= 5000`
- All the integers of `nums` are **unique**.
- `nums` is sorted and rotated between `1` and `n` times.

#### binary search

```python
def findMin(self, nums: List[int]) -> int:
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) >> 1
        if nums[mid] > nums[right]: left = mid + 1
        else: right = mid
    return nums[left]
```

### 162. Find Peak Element

A peak element is an element that is strictly greater than its neighbors.

Given an integer array `nums`, find a peak element, and return its index. If the array contains multiple peaks, return the index to **any of the peaks**.

You may imagine that `nums[-1] = nums[n] = -∞`.

You must write an algorithm that runs in `O(log n)` time.

 

**Example 1:**

```
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.
```

**Example 2:**

```
Input: nums = [1,2,1,3,5,6,4]
Output: 5
Explanation: Your function can return either index number 1 where the peak element is 2, or index number 5 where the peak element is 6.
```

 

**Constraints:**

- `1 <= nums.length <= 1000`
- `-231 <= nums[i] <= 231 - 1`
- `nums[i] != nums[i + 1]` for all valid `i`.

#### binary search

```python
def findPeakElement(self, nums: List[int]) -> int:
    # O(logN) time complexity
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) >> 1
        mid_ = mid + 1
        if nums[mid] > nums[mid_]: right = mid
        else: left = mid_
    return left
```

### 82. Remove Duplicates from Sorted List II

Given the `head` of a sorted linked list, *delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list*. Return *the linked list **sorted** as well*.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/01/04/linkedlist1.jpg)

```
Input: head = [1,2,3,3,4,4,5]
Output: [1,2,5]
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2021/01/04/linkedlist2.jpg)

```
Input: head = [1,1,1,2,3]
Output: [2,3]
```

 

**Constraints:**

- The number of nodes in the list is in the range `[0, 300]`.
- `-100 <= Node.val <= 100`
- The list is guaranteed to be **sorted** in ascending order.

```python
def deleteDuplicates(self, head: ListNode) -> ListNode:
    if not head: return head
    pre = ListNode(-1)
    pre.next = curr = head
    head = pre

    while curr:
        ne = curr.next
        if ne and curr.val == ne.val:
            temp = curr.val
            while curr and curr.val == temp:
                curr = curr.next
            pre.next = curr
        else:
            pre = curr
            curr = curr.next
    return head.next
```

### 15. 3Sum

Given an integer array nums, return all the triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

Notice that the solution set must not contain duplicate triplets.

 

**Example 1:**

```
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
```

**Example 2:**

```
Input: nums = []
Output: []
```

**Example 3:**

```
Input: nums = [0]
Output: []
```

 

**Constraints:**

- `0 <= nums.length <= 3000`
- `-105 <= nums[i] <= 105`

```python
def threeSum(self, nums: List[int]) -> List[List[int]]:
    # if you need to return idx instead of value, use this function
    # idx_nums = sorted(enumerate(nums), key=lambda x: x[1])
    nums.sort()
    ans = list()
    for i in range(len(nums)):
        # prun if this number is the same as the previous one
        # avoid containing duplicate triplets
        if i > 0 and nums[i] == nums[i - 1]: continue
        target = 0 - nums[i]
        left, right = i + 1, len(nums) - 1
        while left < right:
            temp = nums[left] + nums[right]
            if temp == target: 
                ans.append([nums[i], nums[left], nums[right]])
                left_visited = nums[left]
                while left < len(nums) and nums[left] == left_visited:
                    left += 1
            elif temp > target: right -= 1
            else: left += 1
    return ans
```

### 986. Interval List Intersections

You are given two lists of closed intervals, `firstList` and `secondList`, where `firstList[i] = [starti, endi]` and `secondList[j] = [startj, endj]`. Each list of intervals is pairwise **disjoint** and in **sorted order**.

Return *the intersection of these two interval lists*.

A **closed interval** `[a, b]` (with `a <= b`) denotes the set of real numbers `x` with `a <= x <= b`.

The **intersection** of two closed intervals is a set of real numbers that are either empty or represented as a closed interval. For example, the intersection of `[1, 3]` and `[2, 4]` is `[2, 3]`.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2019/01/30/interval1.png)

```
Input: firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
```

**Example 2:**

```
Input: firstList = [[1,3],[5,9]], secondList = []
Output: []
```

**Example 3:**

```
Input: firstList = [], secondList = [[4,8],[10,12]]
Output: []
```

**Example 4:**

```
Input: firstList = [[1,7]], secondList = [[3,10]]
Output: [[3,7]]
```

 

**Constraints:**

- `0 <= firstList.length, secondList.length <= 1000`
- `firstList.length + secondList.length >= 1`
- `0 <= starti < endi <= 109`
- `endi < starti+1`
- `0 <= startj < endj <= 109`
- `endj < startj+1`

```python
def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
    first_idx = second_idx = 0
    m, n = len(firstList), len(secondList)
    ans = list()
    while first_idx < m and second_idx < n:
        f_start, f_end = firstList[first_idx] 
        s_start, s_end = secondList[second_idx]
        if f_end < s_start: first_idx += 1
        elif s_end < f_start: second_idx += 1
        else:
            ans.append([max(f_start, s_start), min(f_end, s_end)])
            if f_end <= s_end: first_idx += 1
            if f_end >= s_end: second_idx += 1
    return ans
```

Runtime: 140 ms, faster than 96.74% of Python3 online submissions for Interval List Intersections.

Memory Usage: 15.1 MB, less than 61.94% of Python3 online submissions for Interval List Intersections.

### 11. Container With Most Water

Given `n` non-negative integers `a1, a2, ..., an` , where each represents a point at coordinate `(i, ai)`. `n` vertical lines are drawn such that the two endpoints of the line `i` is at `(i, ai)` and `(i, 0)`. Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.

**Notice** that you may not slant==倾斜== the container.

 

**Example 1:**

![img](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/17/question_11.jpg)

```
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
```

**Example 2:**

```
Input: height = [1,1]
Output: 1
```

**Example 3:**

```
Input: height = [4,3,2,1,4]
Output: 16
```

**Example 4:**

```
Input: height = [1,2,1]
Output: 2
```

 

**Constraints:**

- `n == height.length`
- `2 <= n <= 105`
- `0 <= height[i] <= 104`

```python
def maxArea(self, height: List[int]) -> int:
    left, right = 0, len(height) - 1
    ans = 0
    while left < right:
        ans = max(ans, min(height[left], height[right]) * (right - left))
        if height[left] > height[right]: right -= 1
        else: left += 1
    return ans
```

### 438. Find All Anagrams in a String

Given two strings `s` and `p`, return *an array of all the start indices of* `p`*'s anagrams in* `s`. You may return the answer in **any order**.

An **Anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

 

**Example 1:**

```
Input: s = "cbaebabacd", p = "abc"
Output: [0,6]
Explanation:
The substring with start index = 0 is "cba", which is an anagram of "abc".
The substring with start index = 6 is "bac", which is an anagram of "abc".
```

**Example 2:**

```
Input: s = "abab", p = "ab"
Output: [0,1,2]
Explanation:
The substring with start index = 0 is "ab", which is an anagram of "ab".
The substring with start index = 1 is "ba", which is an anagram of "ab".
The substring with start index = 2 is "ab", which is an anagram of "ab".
```

 

**Constraints:**

- `1 <= s.length, p.length <= 3 * 104`
- `s` and `p` consist of lowercase English letters.

```python
def findAnagrams(self, s: str, p: str) -> List[int]:
    n = len(p)
    char_count = collections.Counter(s[:n])
    char_count_p = collections.Counter(p)
    ans = list()
    if char_count == char_count_p: ans.append(0)

    for i in range(1, len(s) - n + 1):
        char_count[s[i - 1]] -= 1
        # comparison will take zero values into account
        if char_count[s[i - 1]] == 0: char_count.pop(s[i - 1])
        char_count[s[i + n - 1]] += 1
        if char_count == char_count_p: ans.append(i)
    return ans
```

### 713. Subarray Product Less Than K

Given an array of integers `nums` and an integer `k`, return *the number of contiguous subarrays where the product of all the elements in the subarray is strictly less than* `k`.

 

**Example 1:**

```
Input: nums = [10,5,2,6], k = 100
Output: 8
Explanation: The 8 subarrays that have product less than 100 are:
[10], [5], [2], [6], [10, 5], [5, 2], [2, 6], [5, 2, 6]
Note that [10, 5, 2] is not included as the product of 100 is not strictly less than k.
```

**Example 2:**

```
Input: nums = [1,2,3], k = 0
Output: 0
```

 

**Constraints:**

- `1 <= nums.length <= 3 * 104`
- `1 <= nums[i] <= 1000`
- `0 <= k <= 106`

```python
def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
    # sliding windows
    start = end = ans = 0
    prod = 1
    while end < len(nums):
        prod *= nums[end]
        while start <= end and prod >= k:
            prod /= nums[start]
            start += 1
        # add all subarrays end with nums[end]
        ans += end - start + 1
        end += 1
    return ans
```

### 209. Minimum Size Subarray Sum

Given an array of positive integers `nums` and a positive integer `target`, return the minimal length of a **contiguous subarray** `[numsl, numsl+1, ..., numsr-1, numsr]` of which the sum is greater than or equal to `target`. If there is no such subarray, return `0` instead.

 

**Example 1:**

```
Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: The subarray [4,3] has the minimal length under the problem constraint.
```

**Example 2:**

```
Input: target = 4, nums = [1,4,4]
Output: 1
```

**Example 3:**

```
Input: target = 11, nums = [1,1,1,1,1,1,1,1]
Output: 0
```

 

**Constraints:**

- `1 <= target <= 109`
- `1 <= nums.length <= 105`
- `1 <= nums[i] <= 105`

#### Approach:

Two pointers

```python
def minSubArrayLen(self, target: int, nums: List[int]) -> int:
    n = len(nums)
    left = right = curr_sum = 0
    ans = float('inf')
    while right < n:
        curr_sum += nums[right]
        right += 1
        if curr_sum >= target:
            while curr_sum >= target:
                curr_sum -= nums[left]
                left += 1
            ans = min(ans, right - left + 1)
    return ans if ans != float('inf') else 0
```

### 1091. Shortest Path in Binary Matrix

Given an `n x n` binary matrix `grid`, return *the length of the shortest **clear path** in the matrix*. If there is no clear path, return `-1`.

A **clear path** in a binary matrix is a path from the **top-left** cell (i.e., `(0, 0)`) to the **bottom-right** cell (i.e., `(n - 1, n - 1)`) such that:

- All the visited cells of the path are `0`.
- All the adjacent cells of the path are **8-directionally** connected (i.e., they are different and they share an edge or a corner).

The **length of a clear path** is the number of visited cells of this path.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/02/18/example1_1.png)

```
Input: grid = [[0,1],[1,0]]
Output: 2
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2021/02/18/example2_1.png)

```
Input: grid = [[0,0,0],[1,1,0],[1,1,0]]
Output: 4
```

**Example 3:**

```
Input: grid = [[1,0,0],[1,1,0],[1,1,0]]
Output: -1
```

 

**Constraints:**

- `n == grid.length`
- `n == grid[i].length`
- `1 <= n <= 100`
- `grid[i][j] is 0 or 1`

#### Approach:

BFS

```python
def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
    if grid[0][0] == 1 or grid[-1][-1] == 1: return -1
    n = len(grid)
    dir = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
    visited = set()
    curr_list = [[n - 1, n - 1, 1]] 
    # element: [x_idx, y_idx, step]
    while curr_list:
        x, y, step = curr_list.pop(0)
        if x == 0 and y == 0: return step
        for i, j in dir:
            x_, y_ = x + i, y + j
            if n > x_ >= 0 <= y_ < n and grid[x_][y_] == 0 and (x_, y_) not in visited:
                visited.add((x_, y_))
                curr_list.append([x_, y_, step + 1])
    return -1
```

### 130. Surrounded Regions

Given an `m x n` matrix `board` containing `'X'` and `'O'`, *capture all regions that are 4-directionally surrounded by* `'X'`.

A region is **captured** by flipping all `'O'`s into `'X'`s in that surrounded region.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/02/19/xogrid.jpg)

```
Input: board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
Output: [["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
Explanation: Surrounded regions should not be on the border, which means that any 'O' on the border of the board are not flipped to 'X'. Any 'O' that is not on the border and it is not connected to an 'O' on the border will be flipped to 'X'. Two cells are connected if they are adjacent cells connected horizontally or vertically.
```

**Example 2:**

```
Input: board = [["X"]]
Output: [["X"]]
```

 

**Constraints:**

- `m == board.length`
- `n == board[i].length`
- `1 <= m, n <= 200`
- `board[i][j]` is `'X'` or `'O'`.

#### First Approach:

find all region that is surrounded by X, record all cells, then flip them all.

```python
def solve(self, board: List[List[str]]) -> None:
    """
    Do not return anything, modify board in-place instead.
    """
    m, n = len(board), len(board[0])
    dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    flip_set = set()
    def capture(x, y) -> bool:
        flip_set.add((x, y))
        for i, j in dir:
            x_, y_ = x + i, y + j
            if x_ == m or x_ == -1 or y_ == n or y_ == -1: return False
            if board[x_][y_] == 'O' and (x_, y_) not in flip_set: 
                temp = capture(x_, y_)
                if not temp: return False
        return True

    for i in range(m):
        for j in range(n):
            flip_set.clear()
            if board[i][j] == 'O' and capture(i, j):
                for ii, jj in flip_set:
                    board[ii][jj] = 'X'
```

passed with poor time and space complexity.

#### discussion

- find all `O` cells on boarder, add them to a list
- while this list is not empty, pop one of it, and change this cell to `S`
- traverse board, change all `non-S` cell to `X`, and all `S` to `O`

```python
def solve(self, board: List[List[str]]) -> None:
    """
    Do not return anything, modify board in-place instead.
    """
    m, n = len(board), len(board[0])
    dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    boarder_o = set()
    for i in range(m + n):
        boarder_o |= {(0, i), (m - 1, i), (i, 0), (i, n - 1)}
    while boarder_o:
        i, j = boarder_o.pop()
        if m > i >= 0 <= j < n and board[i][j] == 'O':
            board[i][j] = 'S'
            [boarder_o.add((i + ii, j + jj)) for ii, jj in dir]
    board[:] = [['XO'[i == 'S'] for i in row] for row in board]
```

Runtime: 136 ms, faster than 72.98% of Python3 online submissions for Surrounded Regions.

Memory Usage: 15.4 MB, less than 97.01% of Python3 online submissions for Surrounded Regions.

### 797. All Paths From Source to Target

Given a directed acyclic graph (**DAG**) of `n` nodes labeled from `0` to `n - 1`, find all possible paths from node `0` to node `n - 1` and return them in **any order**.

The graph is given as follows: `graph[i]` is a list of all nodes you can visit from node `i` (i.e., there is a directed edge from node `i` to node `graph[i][j]`).

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/09/28/all_1.jpg)

```
Input: graph = [[1,2],[3],[3],[]]
Output: [[0,1,3],[0,2,3]]
Explanation: There are two paths: 0 -> 1 -> 3 and 0 -> 2 -> 3.
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2020/09/28/all_2.jpg)

```
Input: graph = [[4,3,1],[3,2,4],[3],[4],[]]
Output: [[0,4],[0,3,4],[0,1,3,4],[0,1,2,3,4],[0,1,4]]
```

**Example 3:**

```
Input: graph = [[1],[]]
Output: [[0,1]]
```

**Example 4:**

```
Input: graph = [[1,2,3],[2],[3],[]]
Output: [[0,1,2,3],[0,2,3],[0,3]]
```

**Example 5:**

```
Input: graph = [[1,3],[2],[3],[]]
Output: [[0,1,2,3],[0,3]]
```

 

**Constraints:**

- `n == graph.length`
- `2 <= n <= 15`
- `0 <= graph[i][j] < n`
- `graph[i][j] != i` (i.e., there will be no self-loops).
- All the elements of `graph[i]` are **unique**.
- The input graph is **guaranteed** to be a **DAG**.

#### First Approach:

DFS, recursion

```python
def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
    # DFS
    n = len(graph)
    ans = list()
    def move_next(curr, path):
        if curr == n - 1: 
            ans.append(path.copy())
            return
        for node in graph[curr]:
            path.append(node)
            move_next(node, path)
            path.pop()

    move_next(0, [0])
    return ans
```

### 90. Subsets II

Given an integer array `nums` that may contain duplicates, return *all possible subsets (the power set)*.

The solution set **must not** contain duplicate subsets. Return the solution in **any order**.

 

**Example 1:**

```
Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
```

**Example 2:**

```
Input: nums = [0]
Output: [[],[0]]
```

 

**Constraints:**

- `1 <= nums.length <= 10`
- `-10 <= nums[i] <= 10`

```python
def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
    nums.sort()
    ans = set()
    ans.add(tuple())
    for i in nums:
        new = set()
        for j in ans:
            temp = list(j)
            temp.append(i)
            new.add(tuple(temp))
        ans |= new
    return [list(i) for i in ans]
```

poor time and space complexity

### 78. Subsets

Given an integer array `nums` of **unique** elements, return *all possible subsets (the power set)*.

The solution set **must not** contain duplicate subsets. Return the solution in **any order**.

 

**Example 1:**

```
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

**Example 2:**

```
Input: nums = [0]
Output: [[],[0]]
```

 

**Constraints:**

- `1 <= nums.length <= 10`
- `-10 <= nums[i] <= 10`
- All the numbers of `nums` are **unique**.

```python
def subsets(self, nums: List[int]) -> List[List[int]]:
    nums.sort()
    ans = list()
    ans.append([])
    for i in nums:
        n = len(ans)
        for j in range(n):
            ans.append(ans[j] + [i])
    return ans
```

### 47. Permutations II

Given a collection of numbers, `nums`, that might contain duplicates, return *all possible unique permutations **in any order**.*

 

**Example 1:**

```
Input: nums = [1,1,2]
Output:
[[1,1,2],
 [1,2,1],
 [2,1,1]]
```

**Example 2:**

```
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

 

**Constraints:**

- `1 <= nums.length <= 8`
- `-10 <= nums[i] <= 10`

#### recursion

```python
def permuteUnique(self, nums: List[int]) -> List[List[int]]:
    count = collections.Counter(nums)
    ans = list()
    def find_next(li: list):
        if len(li) == len(nums):
            ans.append(li.copy())
            return
        for key in count.keys():
            if count[key] > 0:
                li.append(key)
                count[key] -= 1
                find_next(li)
                li.pop()
                count[key] += 1
    find_next([])
    return ans
```

### 39. Combination Sum

Given an array of **distinct** integers `candidates` and a target integer `target`, return *a list of all **unique combinations** of* `candidates` *where the chosen numbers sum to* `target`*.* You may return the combinations in **any order**.

The **same** number may be chosen from `candidates` an **unlimited number of times**. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

It is **guaranteed** that the number of unique combinations that sum up to `target` is less than `150` combinations for the given input.

 

**Example 1:**

```
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.
```

**Example 2:**

```
Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]
```

**Example 3:**

```
Input: candidates = [2], target = 1
Output: []
```

**Example 4:**

```
Input: candidates = [1], target = 1
Output: [[1]]
```

**Example 5:**

```
Input: candidates = [1], target = 2
Output: [[1,1]]
```

 

**Constraints:**

- `1 <= candidates.length <= 30`
- `1 <= candidates[i] <= 200`
- All elements of `candidates` are **distinct**.
- `1 <= target <= 500`

```python
def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    ans = list()
    def find_next(pre_sum: int, idx: int, li: list):
        # to avoid making duplicate combinations
        # we cannot choose a number that has a less idx than the previous one
        # which means, our combinations are formed in non-descending idx order
        if pre_sum == target: 
            ans.append(li.copy())
            return
        # ensure there is no duplicate combination
        for i in range(idx, len(candidates)): 
            if pre_sum + candidates[i] <= target:
                li.append(candidates[i])
                find_next(pre_sum + candidates[i], i, li)
                li.pop()
    find_next(0, 0, [])
    return ans
```

Runtime: 67 ms, faster than 80.95% of Python3 online submissions for Combination Sum.

Memory Usage: 14.3 MB, less than 78.31% of Python3 online submissions for Combination Sum.

### 40. Combination Sum II

Given a collection of candidate numbers (`candidates`) and a target number (`target`), find all unique combinations in `candidates` where the candidate numbers sum to `target`.

Each number in `candidates` may only be used **once** in the combination.

**Note:** The solution set must not contain duplicate combinations.

 

**Example 1:**

```
Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
```

**Example 2:**

```
Input: candidates = [2,5,2,1,2], target = 5
Output: 
[
[1,2,2],
[5]
]
```

 

**Constraints:**

- `1 <= candidates.length <= 100`
- `1 <= candidates[i] <= 50`
- `1 <= target <= 30`

```python
def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
    ans = list()
    count = collections.Counter(candidates)
    def find_next(pre_sum: int, li: list):
        if pre_sum == target:
            ans.append(li.copy())
            return
        for key in sorted(count.keys()):
            if pre_sum + key > target: break
            if li and key < li[-1]: continue
            if count[key] <= 0: continue
            li.append(key)
            count[key] -= 1
            find_next(pre_sum + key, li)
            li.pop()
            count[key] += 1
    find_next(0, [])
    return ans
```

### 17. Letter Combinations of a Phone Number

Given a string containing digits from `2-9` inclusive, return all possible letter combinations that the number could represent. Return the answer in **any order**.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Telephone-keypad2.svg/200px-Telephone-keypad2.svg.png)

 

**Example 1:**

```
Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

**Example 2:**

```
Input: digits = ""
Output: []
```

**Example 3:**

```
Input: digits = "2"
Output: ["a","b","c"]
```

 

**Constraints:**

- `0 <= digits.length <= 4`
- `digits[i]` is a digit in the range `['2', '9']`.

```python
def letterCombinations(self, digits: str) -> List[str]:
	# return an empty list rather than list with an empty string if input string is empty
    if not digits: return []
    
    digit_2_letter = {
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz'
    }

    ans = list()
    def append_next(idx, pre):
        if idx == len(digits): ans.append(pre)
        else: [append_next(idx + 1, pre + l) for l in digit_2_letter[digits[idx]]]
    append_next(0, '')
    return ans
```

### 79. Word Search

Given an `m x n` grid of characters `board` and a string `word`, return `true` *if* `word` *exists in the grid*.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/11/04/word2.jpg)

```
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2020/11/04/word-1.jpg)

```
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
Output: true
```

**Example 3:**

![img](https://assets.leetcode.com/uploads/2020/10/15/word3.jpg)

```
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
Output: false
```

 

**Constraints:**

- `m == board.length`
- `n = board[i].length`
- `1 <= m, n <= 6`
- `1 <= word.length <= 15`
- `board` and `word` consists of only lowercase and uppercase English letters.

 

**Follow up:** Could you use search pruning to make your solution faster with a larger `board`?

```python
def exist(self, board: List[List[str]], word: str) -> bool:
    # DFS
    m, n = len(board), len(board[0])
    if len(word) > m * n: return False
    dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    def find_next(x: int, y: int, w: str, visited: set):
        if not w: return True
        for i, j in dir:
            x_, y_ = x + i, y + j
            if m > x_ >= 0 <= y_ < n and board[x_][y_] == w[0] and (x_, y_) not in visited:
                if find_next(x_, y_, w[1:], visited | {(x_, y_)}): return True
        return False

    return any(find_next(i, j, word[1:], {(i, j)}) for i in range(m) for j in range(n) if board[i][j] == word[0])
```

bad performance though..

### 213. House Robber II

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are **arranged in a circle.** That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and **it will automatically contact the police if two adjacent houses were broken into on the same night**.

Given an integer array `nums` representing the amount of money of each house, return *the maximum amount of money you can rob tonight **without alerting the police***.

 

**Example 1:**

```
Input: nums = [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.
```

**Example 2:**

```
Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
```

**Example 3:**

```
Input: nums = [1,2,3]
Output: 3
```

 

**Constraints:**

- `1 <= nums.length <= 100`
- `0 <= nums[i] <= 1000`

```python
def rob_2(self, nums: List[int]) -> int:
    # houses are arranged in a circle
    # which means the first house is adjacent to the last one
    n = len(nums)
    if n == 1: return nums[0]
    # dp[0]: rob first, rob ith
    # dp[1]: rob first, not rob ith
    # dp[2]: not rob first, rob ith
    # dp[3]: not rob first, not rob ith
    dp = [0, nums[0], nums[1], 0]
    for i in range(2, n):
        dp = [dp[1] + nums[i], max(dp[0:2]), dp[3] + nums[i], max(dp[2:4])]
    return max(dp[1:4])
```

### 55. Jump Game

You are given an integer array `nums`. You are initially positioned at the array's **first index**, and each element in the array represents your maximum jump length at that position.

Return `true` *if you can reach the last index, or* `false` *otherwise*.

 

**Example 1:**

```
Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
```

**Example 2:**

```
Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.
```

 

**Constraints:**

- `1 <= nums.length <= 104`
- `0 <= nums[i] <= 105`

Backtracking

```python
def canJump(self, nums: List[int]) -> bool:
    target = len(nums) - 1
    for i in range(len(nums) - 1, -1, -1):
        if nums[i] + i >= target: target = i
        if target == 0: return True
    return False
```

### 62. Unique Paths

A robot is located at the top-left corner of a `m x n` grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2018/10/22/robot_maze.png)

```
Input: m = 3, n = 7
Output: 28
```

**Example 2:**

```
Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Down -> Down
2. Down -> Down -> Right
3. Down -> Right -> Down
```

**Example 3:**

```
Input: m = 7, n = 3
Output: 28
```

**Example 4:**

```
Input: m = 3, n = 3
Output: 6
```

 

**Constraints:**

- `1 <= m, n <= 100`
- It's guaranteed that the answer will be less than or equal to `2 * 109`.

```python
def uniquePaths(self, m: int, n: int) -> int:
    # dp
    dp = [1] * n
    for _ in range(1, m):
        for i in range(1, n):
            dp[i] += dp[i - 1]
    return dp[-1]
```

### 5. Longest Palindromic Substring

Given a string `s`, return *the longest palindromic substring* in `s`.

 

**Example 1:**

```
Input: s = "babad"
Output: "bab"
Note: "aba" is also a valid answer.
```

**Example 2:**

```
Input: s = "cbbd"
Output: "bb"
```

**Example 3:**

```
Input: s = "a"
Output: "a"
```

**Example 4:**

```
Input: s = "ac"
Output: "a"
```

 

**Constraints:**

- `1 <= s.length <= 1000`
- `s` consist of only digits and English letters.

```python
def longestPalindrome(self, s: str) -> str:
    # dp
    n = len(s)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        dp[i][i + 1] = 1
    for step in range(2, n + 1):
        for left in range(n - step):
            if s[left] == s[left + step - 1] and dp[left + 1][left + step - 1] > 0:
                dp[left][left + step] = 2 + dp[left + 1][left + step - 1]
    return max([max(i) for i in dp])
```

### 413. Arithmetic Slices

An integer array is called arithmetic if it consists of **at least three elements** and if the difference between any two consecutive elements is the same.

- For example, `[1,3,5,7,9]`, `[7,7,7,7]`, and `[3,-1,-5,-9]` are arithmetic sequences.

Given an integer array `nums`, return *the number of arithmetic **subarrays** of* `nums`.

A **subarray** is a contiguous subsequence of the array.

 

**Example 1:**

```
Input: nums = [1,2,3,4]
Output: 3
Explanation: We have 3 arithmetic slices in nums: [1, 2, 3], [2, 3, 4] and [1,2,3,4] itself.
```

**Example 2:**

```
Input: nums = [1]
Output: 0
```

 

**Constraints:**

- `1 <= nums.length <= 5000`
- `-1000 <= nums[i] <= 1000`

```python
def numberOfArithmeticSlices(self, nums: List[int]) -> int:
    # contiguous
    if len(nums) < 3: return 0
    ans = 0
    curr = [nums[0]]
    for i in nums[1:]:
        if len(curr) == 1:
            curr.append(i)
        else:
            if i - curr[-1] == curr[-1] - curr[-2]:
                curr.append(i)
                ans += max(0, len(curr) - 2)
            else:
                curr = [curr[-1], i]
    return ans
```

### 201. Bitwise AND of Numbers Range

Given two integers `left` and `right` that represent the range `[left, right]`, return *the bitwise AND of all numbers in this range, inclusive*.

 

**Example 1:**

```
Input: left = 5, right = 7
Output: 4
```

**Example 2:**

```
Input: left = 0, right = 0
Output: 0
```

**Example 3:**

```
Input: left = 1, right = 2147483647
Output: 0
```

 

**Constraints:**

- `0 <= left <= right <= 2 ** 31 - 1`

```python
def rangeBitwiseAnd(self, left: int, right: int) -> int:
    divider, ans = 1, 0
    while divider <= left:
        if left // divider == right // divider and left & divider != 0:
            ans |= divider
        divider <<= 1
    return ans

    # into one-line
    return sum([2 ** expo for expo in range(math.floor(math.log2(left) + 1)) if left // (2 ** expo) == right // (2 ** expo) and left & (2 ** expo) != 0]) if left > 0 else 0

```

### 91. Decode Ways

A message containing letters from `A-Z` can be **encoded** into numbers using the following mapping:

```
'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
```

To **decode** an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, `"11106"` can be mapped into:

- `"AAJF"` with the grouping `(1 1 10 6)`
- `"KJF"` with the grouping `(11 10 6)`

Note that the grouping `(1 11 06)` is invalid because `"06"` cannot be mapped into `'F'` since `"6"` is different from `"06"`.

Given a string `s` containing only digits, return *the **number** of ways to **decode** it*.

The answer is guaranteed to fit in a **32-bit** integer.

 

**Example 1:**

```
Input: s = "12"
Output: 2
Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
```

**Example 2:**

```
Input: s = "226"
Output: 3
Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
```

**Example 3:**

```
Input: s = "0"
Output: 0
Explanation: There is no character that is mapped to a number starting with 0.
The only valid mappings with 0 are 'J' -> "10" and 'T' -> "20", neither of which start with 0.
Hence, there are no valid ways to decode this since all digits need to be mapped.
```

**Example 4:**

```
Input: s = "06"
Output: 0
Explanation: "06" cannot be mapped to "F" because of the leading zero ("6" is different from "06").
```

 

**Constraints:**

- `1 <= s.length <= 100`
- `s` contains only digits and may contain leading zero(s).

```python
def numDecodings(self, s: str) -> int:
    dp = [1, 0, 0, 0]
    for i in s:
        if i == '0':
            dp = [0, 0, 0, dp[1] + dp[2]]
            if dp[3] == 0:
                return 0
        elif i == '1':
            dp = [dp[0] + dp[3], dp[0] + dp[3], 0, dp[1] + dp[2]]
        elif i == '2':
            dp = [dp[0] + dp[3], 0, dp[0] + dp[3], dp[1] + dp[2]]
        elif i in ['3', '4', '5', '6']:
            dp = [dp[0] + dp[3], 0, 0, dp[1] + dp[2]]
        else:
            dp = [dp[0] + dp[3], 0, 0, dp[1]]
    return dp[0] + dp[3]
```

Better in performance ↑↑↑

```python
def numDecodings(self, s: str) -> int:
    # dp = [total ways, ways end with 1, ways end with 2, ways end with 1 digit]
    dp = [1, 0, 0, 0]  
    for c in s:
        curr = int(c)
        # if curr == 0:
        #     dp = [dp[1] + dp[2], 0, 0, 0]
        # elif curr == 1:
        #     dp = [dp[0] + dp[1] + dp[2], dp[0], 0, dp[0]]
        # elif curr == 2:
        #     dp = [dp[0] + dp[1] + dp[2], 0, dp[0], dp[0]]
        # elif curr < 7:
        #     dp = [dp[0] + dp[1] + dp[2], 0, 0, dp[0]]
        # else:
        #     dp = [dp[0] + dp[1], 0, 0, dp[0]]
        dp = [(dp[0] if curr != 0 else 0) + dp[1] + (dp[2] if curr < 7 else 0), dp[0] if curr == 1 else 0, dp[0] if curr == 2 else 0, dp[0] if curr != 0 else 0]
    return dp[0]
```

### 139. Word Break

Given a string `s` and a dictionary of strings `wordDict`, return `true` if `s` can be segmented into a space-separated sequence of one or more dictionary words.

**Note** that the same word in the dictionary may be reused multiple times in the segmentation.

 

**Example 1:**

```
Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
```

**Example 2:**

```
Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.
```

**Example 3:**

```
Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: false
```

 

**Constraints:**

- `1 <= s.length <= 300`
- `1 <= wordDict.length <= 1000`
- `1 <= wordDict[i].length <= 20`
- `s` and `wordDict[i]` consist of only lowercase English letters.
- All the strings of `wordDict` are **unique**.

#### First approach

Use `Trie` class

```python
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    tr = Trie()
    [tr.insert(word) for word in wordDict]

    def find_word(ss: str):
        if not ss: return True
        return any(tr.search(ss[:i + 1]) and find_word(ss[i + 1:]) for i in range(len(ss)))

    return find_word(s)
```

### 394. Decode String

Given an encoded string, return its decoded string.

The encoding rule is: `k[encoded_string]`, where the `encoded_string` inside the square brackets is being repeated exactly `k` times. Note that `k` is guaranteed to be a positive integer.

You may assume that the input string is always valid; No extra white spaces, square brackets are well-formed, etc.

Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, `k`. For example, there won't be input like `3a` or `2[4]`.

 

**Example 1:**

```
Input: s = "3[a]2[bc]"
Output: "aaabcbc"
```

**Example 2:**

```
Input: s = "3[a2[c]]"
Output: "accaccacc"
```

**Example 3:**

```
Input: s = "2[abc]3[cd]ef"
Output: "abcabccdcdcdef"
```

**Example 4:**

```
Input: s = "abc3[cd]xyz"
Output: "abccdcdcdxyz"
```

 

**Constraints:**

- `1 <= s.length <= 30`
- `s` consists of lowercase English letters, digits, and square brackets `'[]'`.
- `s` is guaranteed to be **a valid** input.
- All the integers in `s` are in the range `[1, 300]`.

```python

```

