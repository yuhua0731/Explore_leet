Given a `rows x cols` binary `matrix` filled with `0`'s and `1`'s, find the largest rectangle containing only `1`'s and return *its area*.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/09/14/maximal.jpg)

```
Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 6
Explanation: The maximal rectangle is shown in the above picture.
```

**Example 2:**

```
Input: matrix = []
Output: 0
```

**Example 3:**

```
Input: matrix = [["0"]]
Output: 0
```

**Example 4:**

```
Input: matrix = [["1"]]
Output: 1
```

**Example 5:**

```
Input: matrix = [["0","0"]]
Output: 0
```

 

**Constraints:**

- `rows == matrix.length`
- `cols == matrix[i].length`
- `0 <= row, cols <= 200`
- `matrix[i][j]` is `'0'` or `'1'`.

![image-20211130172015157](/Users/huayu/Library/Application Support/typora-user-images/image-20211130172015157.png)

```python
def maximalRectangle(self, matrix: List[List[str]]) -> int:
    if not matrix: return 0
    m, n = len(matrix), len(matrix[0])
    # for each cell, we need:
    # height of 1's
    # index of first 0 from matrix[i][j] to left
    # index of first 0 from matrix[i][j] to right
    height_left_right = [[[0] * 3 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        lm, rm = -1, n
        for j in range(n):
            if matrix[i][j] == '0':
                lm = j
                continue
            # set height
            height_left_right[i][j][0] = 1 if i == 0 else height_left_right[i - 1][j][0] + 1
            # set left_most
            height_left_right[i][j][1] = lm
        for j in range(n - 1, -1, -1):
            if matrix[i][j] == '0': rm = j
            else: height_left_right[i][j][2] = rm # set right_most
    ans = 0
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '0': continue
            hm, lm, rm = height_left_right[i][j]
            for row in range(i + 1 - hm, i):
                lm = max(lm, height_left_right[row][j][1])
                rm = min(rm, height_left_right[row][j][2])
            ans = max(ans, hm * (rm - lm - 1))
    return ans
```

