### 695. Max Area of Island

You are given an `m x n` binary matrix `grid`. An island is a group of `1`'s (representing land) connected **4-directionally** (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

The **area** of an island is the number of cells with a value `1` in the island.

Return *the maximum **area** of an island in* `grid`. If there is no island, return `0`.

 

**Example 1:**

![img](image_backup/graph theroy/maxarea1-grid.jpg)

```
Input: grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
Output: 6
Explanation: The answer is not 11, because the island must be connected 4-directionally.
```

**Example 2:**

```
Input: grid = [[0,0,0,0,0,0,0,0]]
Output: 0
```

 

**Constraints:**

- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 50`
- `grid[i][j]` is either `0` or `1`.

```python
def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    ans = 0
    m, n = len(grid), len(grid[0])
    dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    def extend(x, y):
        curr = collections.deque([[x, y]])
        grid[x][y] = 0
        cnt = 0
        while curr:
            cnt += 1
            x, y = curr.popleft()
            for i, j in dir:
                xi, yj = x + i, y + j
                if m > xi >= 0 <= yj < n and grid[xi][yj] == 1:
                    grid[xi][yj] = 0
                    curr.append([xi, yj])
        return cnt

    for i, j in product(range(m), range(n)):
        if grid[i][j] == 1:
            ans = max(ans, extend(i, j))
    return ans
```

### 1254. Number of Closed Islands

Given a 2D `grid` consists of `0s` (land) and `1s` (water). An *island* is a maximal 4-directionally connected group of `0s` and a *closed island* is an island **totally** (all left, top, right, bottom) surrounded by `1s.`

Return the number of *closed islands*.

 

**Example 1:**

![img](image_backup/graph theroy/sample_3_1610.png)

```
Input: grid = [[1,1,1,1,1,1,1,0],[1,0,0,0,0,1,1,0],[1,0,1,0,1,1,1,0],[1,0,0,0,0,1,0,1],[1,1,1,1,1,1,1,0]]
Output: 2
Explanation: 
Islands in gray are closed because they are completely surrounded by water (group of 1s).
```

**Example 2:**

![img](image_backup/graph theroy/sample_4_1610.png)

```
Input: grid = [[0,0,1,0,0],[0,1,0,1,0],[0,1,1,1,0]]
Output: 1
```

**Example 3:**

```
Input: grid = [[1,1,1,1,1,1,1],
               [1,0,0,0,0,0,1],
               [1,0,1,1,1,0,1],
               [1,0,1,0,1,0,1],
               [1,0,1,1,1,0,1],
               [1,0,0,0,0,0,1],
               [1,1,1,1,1,1,1]]
Output: 2
```

 

**Constraints:**

- `1 <= grid.length, grid[0].length <= 100`
- `0 <= grid[i][j] <=1`

```python
def closedIsland(self, grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])
    dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    if m == 1 or n == 1: return 0

    def extend(x, y):
        curr = collections.deque([[x, y]])
        grid[x][y] = 1
        while curr:
            x, y = curr.popleft()
            for i, j in dir:
                xi, yj = x + i, y + j
                if m > xi >= 0 <= yj < n and grid[xi][yj] == 0:
                    grid[xi][yj] = 1
                    curr.append([xi, yj])

    for i in range(m):
        if grid[i][0] == 0:
            extend(i, 0)
        if grid[i][n - 1] == 0:
            extend(i, n - 1)
    for i in range(n):
        if grid[0][i] == 0:
            extend(0, i)
        if grid[m - 1][i] == 0:
            extend(m - 1, i)

    ans = 0
    for i, j in product(range(m), range(n)):
        if grid[i][j] == 0:
            ans += 1
            extend(i, j)
    return ans
```

