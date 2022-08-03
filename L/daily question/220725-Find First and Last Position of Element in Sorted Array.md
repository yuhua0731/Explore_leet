Given an array of integers `nums` sorted in non-decreasing order, find the starting and ending position of a given `target` value.

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

- `0 <= nums.length <= 10 ** 5`
- `-10 ** 9 <= nums[i] <= 10 ** 9`
- `nums` is a non-decreasing array.
- `-10 ** 9 <= target <= 10 ** 9`

```python
def searchRange(self, nums: List[int], target: int) -> List[int]:
    # two-point
    # find starting position
    if not nums: return [-1, -1]
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) >> 1
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    if nums[left] != target: return [-1, -1]

    ret = left
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) >> 1
        if nums[mid] <= target:
            left = mid + 1
        else: 
            right = mid
    if nums[left] == target: return [ret, left]
    else: return [ret, left - 1]
```

