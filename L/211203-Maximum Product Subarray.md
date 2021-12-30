Given an integer array `nums`, find a contiguous non-empty subarray within the array that has the largest product, and return *the product*.

It is **guaranteed** that the answer will fit in a **32-bit** integer.

A **subarray** is a contiguous subsequence of the array.

 

**Example 1:**

```
Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
```

**Example 2:**

```
Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
```

 

**Constraints:**

- `1 <= nums.length <= 2 * 104`
- `-10 <= nums[i] <= 10`
- The product of any prefix or suffix of `nums` is **guaranteed** to fit in a **32-bit** integer.

#### DP approach:

```python
def maxProduct(self, nums: List[int]) -> int:
    # avoid incorrect answer if a negative number is the only element in input list
    if len(nums) == 1: return nums[0] 
    # dp[i] = [max_prod positive, max_prod negative]
    # dp[i] represents for the max product of subarray ends in nums[i]
    dp = [[max(0, nums[0]), min(0, nums[0])]]
    for i in nums[1:]:
        pre_posi, pre_nega = dp[-1]
        if i > 0: dp.append([max(pre_posi * i, i), pre_nega * i]) # positive number
        elif i < 0: dp.append([pre_nega * i, min(pre_posi * i, i)]) # negative
        else: dp.append([0, 0]) # zero
    return max([x for x, _ in dp])
```

#### optimization:

dp does not have to be a list, since we only care about the previous one. As for max product, we can update it every loop.

```python
def maxProduct(self, nums: List[int]) -> int:    
    if len(nums) == 1: return nums[0] 
    dp = [max(0, nums[0]), min(0, nums[0])]
    ans = dp[0]
    for i in nums[1:]:
        pre_posi, pre_nega = dp[0], dp[1]
        if i > 0: dp = [max(pre_posi * i, i), pre_nega * i]
        elif i < 0: dp = [pre_nega * i, min(pre_posi * i, i)]
        else: dp = [0, 0]
        ans = max(ans, dp[0])
    return ans
```

Runtime: 56 ms, faster than 68.46% of Python3 online submissions for Maximum Product Subarray.

Memory Usage: 14.2 MB, less than 95.16% of Python3 online submissions for Maximum Product Subarray.