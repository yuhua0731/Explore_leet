An integer array is called arithmetic if it consists of **at least three elements** and if the difference between any two consecutive elements is the same.

- For example, `[1,3,5,7,9]`, `[7,7,7,7]`, and `[3,-1,-5,-9]` are arithmetic sequences.

Given an integer array `nums`, return *the number of arithmetic **subarrays** of* `nums`.

A **subarray** is a ==contiguous== subsequence of the array.

 

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

> - Contiguous subsequence
> - Dp solution: we just need to record the previous state: previous number, and a dict which stores step, size pairs
> - in each loop, we check if current - previous is in previous dict, then generate a new dict for current element

```python
def numberOfArithmeticSlices(self, nums: List[int]) -> int:
    # attention: consecutive elements
    # dp keep records of the valid arithmetic slices ended with current element
    dp = ans = 0
    for i in range(2, len(nums)):
        dp = [0, dp + 1][nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]]
        ans += dp
    return ans
```

