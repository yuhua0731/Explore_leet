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



