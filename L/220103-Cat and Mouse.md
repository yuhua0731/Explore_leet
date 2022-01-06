A game on an **undirected** graph is played by two players, Mouse and Cat, who alternate turns.

The graph is given as follows: `graph[a]` is a list of all nodes `b` such that `ab` is an edge of the graph.

The mouse starts at node `1` and goes first, the cat starts at node `2` and goes second, and there is a hole at node `0`.

During each player's turn, they **must** travel along one edge of the graph that meets where they are. For example, if the Mouse is at node 1, it **must** travel to any node in `graph[1]`.

Additionally, it is not allowed for the Cat to travel to the Hole (node 0.)

Then, the game can end in three ways:

- If ever the Cat occupies the same node as the Mouse, the Cat wins.
- If ever the Mouse reaches the Hole, the Mouse wins.
- If ever a position is repeated (i.e., the players are in the same position as a previous turn, and it is the same player's turn to move), the game is a draw.

Given a `graph`, and assuming both players play ==optimally==, return

- `1` if the mouse wins the game,
- `2` if the cat wins the game, or
- `0` if the game is a draw.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/11/17/cat1.jpg)

```
Input: graph = [[2,5],[3],[0,4,5],[1,4,5],[2,3],[0,2,3]]
Output: 0
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2020/11/17/cat2.jpg)

```
Input: graph = [[1,3],[0],[3],[0,2]]
Output: 1
```

 

**Constraints:**

- `3 <= graph.length <= 50`
- `1 <= graph[i].length < graph.length`
- `0 <= graph[i][j] < graph.length`
- `graph[i][j] != i`
- `graph[i]` is unique.
- The mouse and the cat can always move. 

#### Discussion:

```python
def catMouseGame(self, graph: List[List[int]]) -> int:
    n = len(graph)
    @functools.cache
    def move(step, m, c):
        """
        step: even: mouse turn | odd: cat turn
        m: mouse's position
        c: cat's position
        """
        if step == 2 * n: return 0 # there is no winner after 2n steps, then they will end up draw
        if m == c: return 2 # mouse and cat are in the same position, cat wins
        if m == 0: return 1 # mouse reaches hole, mouse wins

        # move next step
        if step % 2 == 0:
            # mouse turn
            # mouse will take the step optimally
            # once mouse find a chance to win, it will take this step
            if any(move(step + 1, nxt, c) == 1 for nxt in graph[m]): return 1
            # if there is no chance to win, mouse will look for the draw
            if any(move(step + 1, nxt, c) == 0 for nxt in graph[m]): return 0
            # if there is no chance to end with either mouse win or draw, then cat will win
            return 2
        else:
            # cat turn
            if any(move(step + 1, m, nxt) == 2 for nxt in graph[c] if nxt != 0): return 2
            if any(move(step + 1, m, nxt) == 0 for nxt in graph[c] if nxt != 0): return 0
            return 1
    return move(0, 1, 2) # game start with mouse at 1 and cat at 2
```

@cache is essential, it prevents from calculating function with the same input arguments

### 1728. Cat and Mouse II

A game is played by a cat and a mouse named Cat and Mouse.

The environment is represented by a `grid` of size `rows x cols`, where each element is a wall, floor, player (Cat, Mouse), or food.

- Players are represented by the characters `'C'`(Cat)`,'M'`(Mouse).
- Floors are represented by the character `'.'` and can be walked on.
- Walls are represented by the character `'#'` and cannot be walked on.
- Food is represented by the character `'F'` and can be walked on.
- There is only one of each character `'C'`, `'M'`, and `'F'` in `grid`.

Mouse and Cat play according to the following rules:

- Mouse **moves first**, then they take turns to move.
- During each turn, Cat and Mouse can jump in one of the four directions (left, right, up, down). They cannot jump over the wall nor outside of the `grid`.
- `catJump, mouseJump` are the maximum lengths Cat and Mouse can jump at a time, respectively. Cat and Mouse can jump less than the maximum length.
- Staying in the same position is allowed.
- Mouse can jump over Cat.

The game can end in 4 ways:

- If Cat occupies the same position as Mouse, Cat wins.
- If Cat reaches the food first, Cat wins.
- If Mouse reaches the food first, Mouse wins.
- If Mouse cannot get to the food within 1000 turns, Cat wins.

Given a `rows x cols` matrix `grid` and two integers `catJump` and `mouseJump`, return `true` *if Mouse can win the game if both Cat and Mouse play optimally, otherwise return* `false`.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/09/12/sample_111_1955.png)

```
Input: grid = ["####F","#C...","M...."], catJump = 1, mouseJump = 2
Output: true
Explanation: Cat cannot catch Mouse on its turn nor can it get the food before Mouse.
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2020/09/12/sample_2_1955.png)

```
Input: grid = ["M.C...F"], catJump = 1, mouseJump = 4
Output: true
```

**Example 3:**

```
Input: grid = ["M.C...F"], catJump = 1, mouseJump = 3
Output: false
```

 

**Constraints:**

- `rows == grid.length`
- `cols = grid[i].length`
- `1 <= rows, cols <= 8`
- `grid[i][j]` consist only of characters `'C'`, `'M'`, `'F'`, `'.'`, and `'#'`.
- There is only one of each character `'C'`, `'M'`, and `'F'` in `grid`.
- `1 <= catJump, mouseJump <= 8`

```python
def canMouseWin(self, grid: List[str], catJump: int, mouseJump: int) -> bool:
    # allowed steps:
    # 4-directionally, less than or equal to the maximum jump step, without cross the wall
    m, n = len(grid), len(grid[0])
    dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    mx = my = cx = cy = position = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] != '#': position += 1
            if grid[i][j] == 'M': mx, my = i, j
            if grid[i][j] == 'C': cx, cy = i, j

    @functools.cache
    def find_nxt(x, y, jump):
        ans = list()
        for dx, dy in dir:
            for i in range(jump + 1):
                nx, ny = x + dx * i, y + dy * i
                if m > nx >= 0 <= ny < n and grid[nx][ny] != '#': 
                    ans.append([nx, ny])
                else:
                    break
        return ans

    @functools.cache
    def move(step, mx, my, cx, cy):
        if step > position * 2: return False
        if mx == cx and my == cy: return False
        if grid[mx][my] == 'F': return True
        if grid[cx][cy] == 'F': return False

        if step % 2 == 0:
            # mouse turn
            return True if any(move(step + 1, x, y, cx, cy) for x, y in find_nxt(mx, my, mouseJump)) else False
        else:
            # cat turn
            return False if any(not move(step + 1, mx, my, x, y) for x, y in find_nxt(cx, cy, catJump)) else True


    return move(0, mx, my, cx, cy)
```

I got TLE when I use the constraint specified in question that mouse loses game if step exceeds 1000.

Actually, we can derive from the previous question that if step reaches 2n(n is the available ceil in the map), the game will be a draw(cat wins in this question). Hence, we can decrease recursive call depth to 2 * position from 1000, which will significantly improve our time complexity.