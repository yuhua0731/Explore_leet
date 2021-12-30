Given a **non-empty** array `nums` containing **only positive integers**, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.

 

**Example 1:**

```
Input: nums = [1,5,11,5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].
```

**Example 2:**

```
Input: nums = [1,2,3,5]
Output: false
Explanation: The array cannot be partitioned into equal sum subsets.
```

 

**Constraints:**

- `1 <= nums.length <= 200`
- `1 <= nums[i] <= 100`

#### First Approach:

DFS, recursively call

traverse all possible combinations, prun when sum exceed target.

```python
def canPartition(self, nums: List[int]) -> bool:
    if sum(nums) % 2 != 0: return False
    target = sum(nums) // 2

    def add_next(idx, pre_sum):
        if pre_sum > target: return False
        if pre_sum == target: return True
        return any(add_next(i, pre_sum + nums[i]) for i in range(idx + 1, len(nums)))

    return add_next(-1, 0)
```

result in TLE

#### Discussion:

DP approach

`dp = [False] * (target + 1)`

dp[i] represents for the result of target = i

```python
def canPartition(self, nums: List[int]) -> bool:
    # dp
    target, n = sum(nums), len(nums)
    if target & 1: return False
    target >>= 1
    dp = [True] + [False] * target

    for i in nums:
        dp = [dp[s] or (s >= i and dp[s - i]) for s in range(target + 1)]
        if dp[-1]: return True
    return False
```

