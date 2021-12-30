We have `n` chips, where the position of the `ith` chip is `position[i]`.

We need to move all the chips to **the same position**. In one step, we can change the position of the `ith` chip from `position[i]` to:

- `position[i] + 2` or `position[i] - 2` with `cost = 0`.
- `position[i] + 1` or `position[i] - 1` with `cost = 1`.

Return *the minimum cost* needed to move all the chips to the same position.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/08/15/chips_e1.jpg)

```
Input: position = [1,2,3]
Output: 1
Explanation: First step: Move the chip at position 3 to position 1 with cost = 0.
Second step: Move the chip at position 2 to position 1 with cost = 1.
Total cost is 1.
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2020/08/15/chip_e2.jpg)

```
Input: position = [2,2,2,3,3]
Output: 2
Explanation: We can move the two chips at position  3 to position 2. Each move has cost = 1. The total cost = 2.
```

**Example 3:**

```
Input: position = [1,1000000000]
Output: 1
```

 

**Constraints:**

- `1 <= position.length <= 100`
- `1 <= position[i] <= 10^9`

#### Approach:

Since it cost 0 to move 2 steps. We can gather all even chips and all odd chips without any cost.

Then, the last action is choose the less one(between even group and odd group), ans move all chips of it to the other one.

```python
def minCostToMoveChips(self, position: List[int]) -> int:
    ans = [0] * 2 # even, odd
    for i in position: ans[i % 2] += 1
    return min(ans)
```

