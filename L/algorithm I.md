### 70. Climbing Stairs

You are climbing a staircase. It takes `n` steps to reach the top.

Each time you can either climb `1` or `2` steps. In how many distinct ways can you climb to the top?

```python
def climbStairs(self, n: int) -> int:
    # dp
    pre_2, pre_1 = 0, 1
    for _ in range(1, n + 1):
        curr = pre_2 + pre_1
        pre_2, pre_1 = pre_1, curr
    return pre_1
```

### 198. House Robber

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and **it will automatically contact the police if two adjacent houses were broken into on the same night**.

Given an integer array `nums` representing the amount of money of each house, return *the maximum amount of money you can rob tonight **without alerting the police***.

```python
def rob(self, nums: List[int]) -> int:
    # why do we need to record up to 3 steps previously?
    # consider this condition: 2, 1, 1, 2
    # the max money we can rob is 2 + 2
    pre = [0] * 3
    for i in nums:
        pre = [pre[1], pre[2], max(pre[0], pre[1]) + i]
    return max(pre)
```

### 120. Triangle

Given a `triangle` array, return *the minimum path sum from top to bottom*.

For each step, you may move to an adjacent number of the row below. More formally, if you are on index `i` on the current row, you may move to either index `i` or index `i + 1` on the next row.

```python
def minimumTotal(self, triangle: List[List[int]]) -> int:
    # dp
    # first row, remain original
    n = len(triangle)
    for i in range(1, n):
        for j in range(i + 1):
            """
            for ith row, we have i + 1 elements. indexed from 0 to i
            for previous row, we have i elements, indexed from 0 to i - 1
            for element [i][j], we should compare [i - 1][j - 1] & [i - 1][j]
            """
            triangle[i][j] += min([triangle[i - 1][k] for k in range(max(0, j - 1), min(i, j + 1))])
    return min(triangle[-1])
```

