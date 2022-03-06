Given an `m x n` binary `matrix` filled with `0`'s and `1`'s, *find the largest square containing only* `1`'s *and return its area*.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/11/26/max1grid.jpg)

```
Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 4
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2020/11/26/max2grid.jpg)

```
Input: matrix = [["0","1"],["1","0"]]
Output: 1
```

**Example 3:**

```
Input: matrix = [["0"]]
Output: 0
```

 

**Constraints:**

- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= m, n <= 300`
- `matrix[i][j]` is `'0'` or `'1'`.

```python
def maximalSquare(self, matrix: List[List[str]]) -> int:
    # square, much easier
    # dp
    dp = [int(i) for i in matrix[0]]
    ans = max(dp)
    for row in matrix[1:]:
        temp = [int(row[0])]
        for idx in range(1, len(row)):
            temp.append(0 if row[idx] == '0' else 1 + min([temp[-1], dp[idx - 1], dp[idx]]))
        ans = max(ans, max(temp))
        dp = temp
    return ans ** 2
```

