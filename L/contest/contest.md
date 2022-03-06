# 2127. Maximum Employees to Be Invited to a Meeting

### weekly contest 274

A company is organizing a meeting and has a list of `n` employees, waiting to be invited. They have arranged for a large **circular** table, capable of seating **any number** of employees.

The employees are numbered from `0` to `n - 1`. Each employee has a **favorite** person and they will attend the meeting **only if** they can sit next to their favorite person at the table. The favorite person of an employee is **not** themself.

Given a **0-indexed** integer array `favorite`, where `favorite[i]` denotes the favorite person of the `ith` employee, return *the **maximum number of employees** that can be invited to the meeting*.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/12/14/ex1.png)

```
Input: favorite = [2,2,1,2]
Output: 3
Explanation:
The above figure shows how the company can invite employees 0, 1, and 2, and seat them at the round table.
All employees cannot be invited because employee 2 cannot sit beside employees 0, 1, and 3, simultaneously.
Note that the company can also invite employees 1, 2, and 3, and give them their desired seats.
The maximum number of employees that can be invited to the meeting is 3. 
```

**Example 2:**

```
Input: favorite = [1,2,0]
Output: 3
Explanation: 
Each employee is the favorite person of at least one other employee, and the only way the company can invite them is if they invite every employee.
The seating arrangement will be the same as that in the figure given in example 1:
- Employee 0 will sit between employees 2 and 1.
- Employee 1 will sit between employees 0 and 2.
- Employee 2 will sit between employees 1 and 0.
The maximum number of employees that can be invited to the meeting is 3.
```

**Example 3:**

![img](https://assets.leetcode.com/uploads/2021/12/14/ex2.png)

```
Input: favorite = [3,0,1,4,1]
Output: 4
Explanation:
The above figure shows how the company will invite employees 0, 1, 3, and 4, and seat them at the round table.
Employee 2 cannot be invited because the two spots next to their favorite employee 0 are taken.
So the company leaves them out of the meeting.
The maximum number of employees that can be invited to the meeting is 4.
```

 

**Constraints:**

- `n == favorite.length`
- `2 <= n <= 10 ** 5`
- `0 <= favorite[i] <= n - 1`
- `favorite[i] != i`

Failed to solve this problem during contest. :(

After viewing some excellent posts in discussion board, I have learned a solution and implemented a version of mine.

![image](https://assets.leetcode.com/users/images/727de57e-a6b4-4ee3-b714-9bcee1234704_1641163766.1036708.png)

> - there are two condisions that we need to take care:
>   - Case 1 - if some employees can form a cycle, then they can be fit in a table, while no other people can join in.
>   - Case 2 - if there are two employees that fall in love with each other, we call them a ==pair==. Apart from the previous cycle we have mentioned above, these 2-people pairs allow more people to join in. For instance, A & B is a pair, C can still sit adjacent to A if C likes A. Thus, for each pair, we should extend two longest arms from two employees. Y -> U -> I -> A <=> B <- S <- P <- Q <- K is a pair with two longest arms from A & B, and the size of it is 4 + 5 = 9.
>
> - for case 1, we can only pick up the cycle with the greatest amount of employees as our answer; for case 2, we can pick up all pairs with their longest arms, and sum them up as our answer
> - choose the bigger answer amount case 1 and case 2.

![image](https://assets.leetcode.com/users/images/9d073878-2b25-4aa6-b3ca-6c74f88efdc2_1641165402.3325021.png)

```python
def maximumInvitations(self, favorite: List[int]) -> int:
    # case 1: 寻找最大的环， size >= 3
    # case 2: 找到所有互相喜欢的员工（pairs），以两者为起点，反向延伸找到最长的被喜欢链len = a & b
    # sum(a + b for all pairs)
    # return max(case 1, case 2)

    n = len(favorite)
    cycles = list()

    # form a be-liked dict to search employee backwards
    liked = collections.defaultdict(list)
    for i, j in enumerate(favorite):
        liked[j].append(i)

    # case 1
    for i in range(n):
        if favorite[i] == -1:
            continue
        path = {i: 0}
        while favorite[i] != -1:
            temp = favorite[i]
            favorite[i] = -1
            if temp in path:
                cycles.append([temp, len(path) - path[temp]])
                break
            i = temp
            path[temp] = len(path)

    # case 2
    # we already detect all pairs in case 1
    def extend(idx, exclude):
        pre = liked[idx]
        ans = 0
        for pre in liked[idx]:
            if pre != exclude:
                ans = max(ans, extend(pre, exclude))
        return ans + 1

    max_case1, max_case2 = 0, 0
    for i, size in cycles:
        if size > 2:
            max_case1 = max(max_case1, size)
        else:
            j = [temp for temp in liked[i] if i in liked[temp]][0]
            max_case2 += extend(i, j) + extend(j, i)
    return max(max_case1, max_case2)
```



# 2132. Stamping the Grid

### biweekly contest 69

You are given an `m x n` binary matrix `grid` where each cell is either `0` (empty) or `1` (occupied).

You are then given stamps of size `stampHeight x stampWidth`. We want to fit the stamps such that they follow the given **restrictions** and **requirements**:

1. Cover all the **empty** cells.
2. Do not cover any of the **occupied** cells.
3. We can put as **many** stamps as we want.
4. Stamps can **overlap** with each other.
5. Stamps are not allowed to be **rotated**.
6. Stamps must stay completely **inside** the grid.

Return `true` *if it is possible to fit the stamps while following the given restrictions and requirements. Otherwise, return* `false`.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/11/03/ex1.png)

```
Input: grid = [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]], stampHeight = 4, stampWidth = 3
Output: true
Explanation: We have two overlapping stamps (labeled 1 and 2 in the image) that are able to cover all the empty cells.
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2021/11/03/ex2.png)

```
Input: grid = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], stampHeight = 2, stampWidth = 2 
Output: false 
Explanation: There is no way to fit the stamps onto all the empty cells without the stamps going outside the grid.
```

 

**Constraints:**

- `m == grid.length`
- `n == grid[r].length`
- `1 <= m, n <= 10 ** 5`
- `1 <= m * n <= 2 * 10 ** 5`
- `grid[r][c]` is either `0` or `1`.
- `1 <= stampHeight, stampWidth <= 10 ** 5`

#### First, go through this pre-question:

##### 304. Range Sum Query 2D - Immutable

Given a 2D matrix `matrix`, handle multiple queries of the following type:

- Calculate the **sum** of the elements of `matrix` inside the rectangle defined by its **upper left corner** `(row1, col1)` and **lower right corner** `(row2, col2)`.

Implement the NumMatrix class:

- `NumMatrix(int[][] matrix)` Initializes the object with the integer matrix `matrix`.
- `int sumRegion(int row1, int col1, int row2, int col2)` Returns the **sum** of the elements of `matrix` inside the rectangle defined by its **upper left corner** `(row1, col1)` and **lower right corner** `(row2, col2)`.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/03/14/sum-grid.jpg)

```
Input
["NumMatrix", "sumRegion", "sumRegion", "sumRegion"]
[[[[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]]], [2, 1, 4, 3], [1, 1, 2, 2], [1, 2, 2, 4]]
Output
[null, 8, 11, 12]

Explanation
NumMatrix numMatrix = new NumMatrix([[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]]);
numMatrix.sumRegion(2, 1, 4, 3); // return 8 (i.e sum of the red rectangle)
numMatrix.sumRegion(1, 1, 2, 2); // return 11 (i.e sum of the green rectangle)
numMatrix.sumRegion(1, 2, 2, 4); // return 12 (i.e sum of the blue rectangle)
```

 

**Constraints:**

- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= m, n <= 200`
- `-10 ** 5 <= matrix[i][j] <= 10 ** 5 `
- `0 <= row1 <= row2 < m`
- `0 <= col1 <= col2 < n`
- At most `10 ** 4` calls will be made to `sumRegion`.

```python
#!/usr/bin/env python3
from typing import List

class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.m = matrix
        m, n = len(matrix), len(matrix[0])
        for i in range(m):
            for j in range(n):
                self.m[i][j] += (self.m[i - 1][j] if i > 0 else 0) + (self.m[i][j - 1] if j > 0 else 0) - (self.m[i - 1][j - 1] if i > 0 and j > 0 else 0)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.m[row2][col2] - (self.m[row1 - 1][col2] if row1 > 0 else 0) - (self.m[row2][col1 - 1] if col1 > 0 else 0) + (self.m[row1 - 1][col1 - 1] if row1 > 0 and col1 > 0 else 0)
```

1. convert original input matrix into 2d cumulative sums. `matrix[i][j]` represents for the **sum** of the elements of `matrix` inside the rectangle defined by its **upper left corner** `(0, 0)` and **lower right corner** `(i, j)`.
2. derive sumRegion from 4 corners’ value

#### Now, let’s go back to out original problem:

- Let the height and width of the stamp be `h` and `w`, and the height and width of the grid be `H` and `W`.

- First, go through every **empty** cell and check if it can be as the right bottom corner of a stamp.

  - Let the cell we are checking be `(x, y)`

    (The list is 0-based). Then it can be a valid right bottom corner of a stamp if the following requirements are satisfied:

    - `x >= h - 1 and y >= w - 1`. Otherwise, the stamp may be pasted outside the grid.

    - There are no occupied cells in `grid[x-h+1][y-w+1] ~ grid[x][y]`.

      > to check if all cells are empty in `grid[x-h+1][y-w+1] ~ grid[x][y]`, we can refer to the above question 304. If subRegion has a sum of 0, then all cells are empty.

    - For example, if `h = w = 2` then the valid right bottom corners will look like:
      ![img](https://i.imgur.com/1DyDrz6.png)

- Second, go through the **empty** cells and check if they are all covered by stamps.

  - Let the cell we are checking be `(x, y)`. Then the cell is covered by stamps if there are valid bottom right corners in `grid[x][y] ~ grid[min(x+h-1, H)][min(y+w-1, W)]`.
  - For example, if we are checking the cell `(0, 2)` (the cell with orange triangle) then it is covered by stamps placed on the cell `(1, 2)` and `(1, 3)`.
    ![img](https://i.imgur.com/a04r2IK.png)

```python
def possibleToStamp(self, grid: List[List[int]], stampHeight: int, stampWidth: int) -> bool:
    m, n = len(grid), len(grid[0])
    H, W = stampHeight, stampWidth
    def acc_2d(grid):
        dp = [[0] * (n + 1) for _ in range(m + 1)] 
        for c, r in product(range(n), range(m)):
            dp[r + 1][c + 1] = dp[r + 1][c] + dp[r][c + 1] - dp[r][c] + grid[r][c]
        return dp

    def sumRegion(mat, r1, c1, r2, c2):
        return mat[r2 + 1][c2 + 1] - mat[r1][c2 + 1] - mat[r2 + 1][c1] + mat[r1][c1]  

    dp = acc_2d(grid)
    stamp_grid = [[0] * (n + 1) for _ in range(m + 1)] 
    for r, c in product(range(m - H + 1), range(n - W + 1)):
        if sumRegion(dp, r, c, r + H - 1, c + W - 1) == 0:
            # all cells in this range are empty
            # just mark the right-bottom corner cell with 1
            stamp_grid[r + H][c + W] = 1
	
    stamp_prefix = acc_2d(stamp_grid)
    for r, c in product(range(m - H + 1), range(n - W + 1)):
        # cell is empty and cannot be a right-bottom corner of a stamp
        if grid[r][c] == 0 and stamp_grid[r][c] == 0:
            if sumRegion(stamp_prefix, r, c, r + H - 1, c + W - 1) == 0:
                # this cell cannot be covered by any right-bottom corner
                return False
    return True
	
```

# 2141. Maximum Running Time of N Computers

### weekly contest 276

You have `n` computers. You are given the integer `n` and a **0-indexed** integer array `batteries` where the `ith` battery can **run** a computer for `batteries[i]` minutes. You are interested in running **all** `n` computers **simultaneously** using the given batteries.

Initially, you can insert **at most one battery** into each computer. After that and at any integer time moment, you can remove a battery from a computer and insert another battery **any number of times**. The inserted battery can be a totally new battery or a battery from another computer. You may assume that the removing and inserting processes take no time.

Note that the batteries cannot be recharged.

Return *the **maximum** number of minutes you can run all the* `n` *computers simultaneously.*

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2022/01/06/example1-fit.png)

```
Input: n = 2, batteries = [3,3,3]
Output: 4
Explanation: 
Initially, insert battery 0 into the first computer and battery 1 into the second computer.
After two minutes, remove battery 1 from the second computer and insert battery 2 instead. Note that battery 1 can still run for one minute.
At the end of the third minute, battery 0 is drained, and you need to remove it from the first computer and insert battery 1 instead.
By the end of the fourth minute, battery 1 is also drained, and the first computer is no longer running.
We can run the two computers simultaneously for at most 4 minutes, so we return 4.
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2022/01/06/example2.png)

```
Input: n = 2, batteries = [1,1,1,1]
Output: 2
Explanation: 
Initially, insert battery 0 into the first computer and battery 2 into the second computer. 
After one minute, battery 0 and battery 2 are drained so you need to remove them and insert battery 1 into the first computer and battery 3 into the second computer. 
After another minute, battery 1 and battery 3 are also drained so the first and second computers are no longer running.
We can run the two computers simultaneously for at most 2 minutes, so we return 2.
```

 

**Constraints:**

- `1 <= n <= batteries.length <= 105`
- `1 <= batteries[i] <= 109`

#### My approach

After 3 times of TLE and 1 time of WA, I finally got accepted for this question during the contest.

this is my intuition at the beginning:

> - the upper bound of our answer should be decided as `maxAns = sum(batteries) // n`
> - if there is any battery that have greater capacity than maxAns, then that amount will be wasted. since this one battery can only be applied to one computer, and satisfy any answer less than maxAns.
> - Hence, we can lower our question to n = n - 1, batteries = sorted(batteries)[:-1]

```python
def maxRunTime(self, n: int, batteries: List[int]) -> int:
    batteries.sort()
    total = sum(batteries)
    while batteries[-1] > total // n:
        n -= 1
        total -= batteries.pop()
    return total // n
```

Q4 is mainly focus on your intuition, rather than your code ability

[Lee’s intuition](https://leetcode.com/problems/maximum-running-time-of-n-computers/discuss/1692939/JavaC%2B%2BPython-Sort-Solution-with-Explanation-O(mlogm))



# 2151. Maximum Good People Based on Statements

### weekly contest 277

There are two types of persons:

- The **good person**: The person who always tells the truth.
- The **bad person**: The person who might tell the truth and might lie.

You are given a **0-indexed** 2D integer array `statements` of size `n x n` that represents the statements made by `n` people about each other. More specifically, `statements[i][j]` could be one of the following:

- `0` which represents a statement made by person `i` that person `j` is a **bad** person.
- `1` which represents a statement made by person `i` that person `j` is a **good** person.
- `2` represents that **no statement** is made by person `i` about person `j`.

Additionally, no person ever makes a statement about themselves. Formally, we have that `statements[i][i] = 2` for all `0 <= i < n`.

Return *the **maximum** number of people who can be **good** based on the statements made by the* `n` *people*.

 

**Example 1:**

![img](image_backup/contest/logic1.jpg)

```
Input: statements = [[2,1,2],[1,2,2],[2,0,2]]
Output: 2
Explanation: Each person makes a single statement.
- Person 0 states that person 1 is good.
- Person 1 states that person 0 is good.
- Person 2 states that person 1 is bad.
Let's take person 2 as the key.
- Assuming that person 2 is a good person:
    - Based on the statement made by person 2, person 1 is a bad person.
    - Now we know for sure that person 1 is bad and person 2 is good.
    - Based on the statement made by person 1, and since person 1 is bad, they could be:
        - telling the truth. There will be a contradiction in this case and this assumption is invalid.
        - lying. In this case, person 0 is also a bad person and lied in their statement.
    - Following that person 2 is a good person, there will be only one good person in the group.
- Assuming that person 2 is a bad person:
    - Based on the statement made by person 2, and since person 2 is bad, they could be:
        - telling the truth. Following this scenario, person 0 and 1 are both bad as explained before.
            - Following that person 2 is bad but told the truth, there will be no good persons in the group.
        - lying. In this case person 1 is a good person.
            - Since person 1 is a good person, person 0 is also a good person.
            - Following that person 2 is bad and lied, there will be two good persons in the group.
We can see that at most 2 persons are good in the best case, so we return 2.
Note that there is more than one way to arrive at this conclusion.
```

**Example 2:**

![img](image_backup/contest/logic2.jpg)

```
Input: statements = [[2,0],[0,2]]
Output: 1
Explanation: Each person makes a single statement.
- Person 0 states that person 1 is bad.
- Person 1 states that person 0 is bad.
Let's take person 0 as the key.
- Assuming that person 0 is a good person:
    - Based on the statement made by person 0, person 1 is a bad person and was lying.
    - Following that person 0 is a good person, there will be only one good person in the group.
- Assuming that person 0 is a bad person:
    - Based on the statement made by person 0, and since person 0 is bad, they could be:
        - telling the truth. Following this scenario, person 0 and 1 are both bad.
            - Following that person 0 is bad but told the truth, there will be no good persons in the group.
        - lying. In this case person 1 is a good person.
            - Following that person 0 is bad and lied, there will be only one good person in the group.
We can see that at most, one person is good in the best case, so we return 1.
Note that there is more than one way to arrive at this conclusion.
```

 

**Constraints:**

- `n == statements.length == statements[i].length`
- `2 <= n <= 15`
- `statements[i][j]` is either `0`, `1`, or `2`.
- `statements[i][i] == 2`

failed to solve this question during contest.

afraid of getting TLE, didn’t try brute force..



1. since 1 <= n <= 15, this is a small size problem even we implement brute force solution.
2. using bitmask to represent that each person is good or bad.
3. implement a function to check if a bitmask is valid.
4. Keep track of the maximum amount of good persons.

```python
def maximumGood(self, statements: List[List[int]]) -> int:
    ans = 0
    n = len(statements)
    def check(b):
        for i, p in enumerate(b):
            if p == '1': 
                # person i is good person
                for idx, s in enumerate(statements[i]):
                    # s == 0 && b[idx] = 0
                    # s == 1 && b[idx] = 1
                    # s == 2
                    if s != 2 and s != int(b[idx]): return False
        return True

    for i in range(2 ** n):
        bitmask = bin(i)[2:].zfill(n)
        if check(bitmask):
            ans = max(ans, bitmask.count('1'))
    return ans
```

Runtime: 1475 ms, faster than 70.00% of Python3 online submissions for Maximum Good People Based on Statements.

Memory Usage: 14.2 MB, less than 100.00% of Python3 online submissions for Maximum Good People Based on Statements.

such a simple question, you really should think more straight.
