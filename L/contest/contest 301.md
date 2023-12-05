### 2335. Minimum Amount of Time to Fill Cups

You have a water dispenser that can dispense cold, warm, and hot water. Every second, you can either fill up `2` cups with **different** types of water, or `1` cup of any type of water.

You are given a **0-indexed** integer array `amount` of length `3` where `amount[0]`, `amount[1]`, and `amount[2]` denote the number of cold, warm, and hot water cups you need to fill respectively. Return *the **minimum** number of seconds needed to fill up all the cups*.

 

**Example 1:**

```
Input: amount = [1,4,2]
Output: 4
Explanation: One way to fill up the cups is:
Second 1: Fill up a cold cup and a warm cup.
Second 2: Fill up a warm cup and a hot cup.
Second 3: Fill up a warm cup and a hot cup.
Second 4: Fill up a warm cup.
It can be proven that 4 is the minimum number of seconds needed.
```

**Example 2:**

```
Input: amount = [5,4,4]
Output: 7
Explanation: One way to fill up the cups is:
Second 1: Fill up a cold cup, and a hot cup.
Second 2: Fill up a cold cup, and a warm cup.
Second 3: Fill up a cold cup, and a warm cup.
Second 4: Fill up a warm cup, and a hot cup.
Second 5: Fill up a cold cup, and a hot cup.
Second 6: Fill up a cold cup, and a warm cup.
Second 7: Fill up a hot cup.
```

**Example 3:**

```
Input: amount = [5,0,0]
Output: 5
Explanation: Every second, we fill up a cold cup.
```

 

**Constraints:**

- `amount.length == 3`
- `0 <= amount[i] <= 100`

```python
def fillCups(self, amount: List[int]) -> int:
    amount.sort()
    if sum(amount[:2]) <= amount[-1]: return amount[-1]
    return math.ceil((sum(amount[:2]) - amount[-1]) / 2) + amount[-1]
```



### 2336. Smallest Number in Infinite Set

You have a set which contains all positive integers `[1, 2, 3, 4, 5, ...]`.

Implement the `SmallestInfiniteSet` class:

- `SmallestInfiniteSet()` Initializes the **SmallestInfiniteSet** object to contain **all** positive integers.
- `int popSmallest()` **Removes** and returns the smallest integer contained in the infinite set.
- `void addBack(int num)` **Adds** a positive integer `num` back into the infinite set, if it is **not** already in the infinite set.

 

**Example 1:**

```
Input
["SmallestInfiniteSet", "addBack", "popSmallest", "popSmallest", "popSmallest", "addBack", "popSmallest", "popSmallest", "popSmallest"]
[[], [2], [], [], [], [1], [], [], []]
Output
[null, null, 1, 2, 3, null, 1, 4, 5]

Explanation
SmallestInfiniteSet smallestInfiniteSet = new SmallestInfiniteSet();
smallestInfiniteSet.addBack(2);    // 2 is already in the set, so no change is made.
smallestInfiniteSet.popSmallest(); // return 1, since 1 is the smallest number, and remove it from the set.
smallestInfiniteSet.popSmallest(); // return 2, and remove it from the set.
smallestInfiniteSet.popSmallest(); // return 3, and remove it from the set.
smallestInfiniteSet.addBack(1);    // 1 is added back to the set.
smallestInfiniteSet.popSmallest(); // return 1, since 1 was added back to the set and
                                   // is the smallest number, and remove it from the set.
smallestInfiniteSet.popSmallest(); // return 4, and remove it from the set.
smallestInfiniteSet.popSmallest(); // return 5, and remove it from the set.
```

 

**Constraints:**

- `1 <= num <= 1000`
- At most `1000` calls will be made **in total** to `popSmallest` and `addBack`.

```python
class SmallestInfiniteSet:

    def __init__(self):
        self.num = [True] * 1000
        self.s = [i + 1 for i in range(1000)]
        heapq.heapify(self.s)

    def popSmallest(self) -> int:
        while self.s and not self.num[self.s[0] - 1]:
            heapq.heappop(self.s)
        ret = heapq.heappop(self.s)
        self.num[ret - 1] = False
        return ret

    def addBack(self, num: int) -> None:
        if self.num[num - 1]: return
        self.num[num - 1] = True
        heapq.heappush(self.s, num)

# Your SmallestInfiniteSet object will be instantiated and called as such:
# obj = SmallestInfiniteSet()
# param_1 = obj.popSmallest()
# obj.addBack(num)
```



### 2337. Move Pieces to Obtain a String

You are given two strings `start` and `target`, both of length `n`. Each string consists **only** of the characters `'L'`, `'R'`, and `'_'` where:

- The characters `'L'` and `'R'` represent pieces, where a piece `'L'` can move to the **left** only if there is a **blank** space directly to its left, and a piece `'R'` can move to the **right** only if there is a **blank** space directly to its right.
- The character `'_'` represents a blank space that can be occupied by **any** of the `'L'` or `'R'` pieces.

Return `true` *if it is possible to obtain the string* `target` *by moving the pieces of the string* `start` ***any** number of times*. Otherwise, return `false`.

 

**Example 1:**

```
Input: start = "_L__R__R_", target = "L______RR"
Output: true
Explanation: We can obtain the string target from start by doing the following moves:
- Move the first piece one step to the left, start becomes equal to "L___R__R_".
- Move the last piece one step to the right, start becomes equal to "L___R___R".
- Move the second piece three steps to the right, start becomes equal to "L______RR".
Since it is possible to get the string target from start, we return true.
```

**Example 2:**

```
Input: start = "R_L_", target = "__LR"
Output: false
Explanation: The 'R' piece in the string start can move one step to the right to obtain "_RL_".
After that, no pieces can move anymore, so it is impossible to obtain the string target from start.
```

**Example 3:**

```
Input: start = "_R", target = "R_"
Output: false
Explanation: The piece in the string start can move only to the right, so it is impossible to obtain the string target from start.
```

 

**Constraints:**

- `n == start.length == target.length`
- `1 <= n <= 10 ** 5`
- `start` and `target` consist of the characters `'L'`, `'R'`, and `'_'`.

```python
def canChange(self, start: str, target: str) -> bool:
    if start.replace('_', '') != target.replace('_', ''): return False
    start_L, start_R = [], []
    target_L, target_R = [], []
    for idx, i in enumerate(start):
        if i == 'L': start_L.append(idx)
        if i == 'R': start_R.append(idx)
    for idx, i in enumerate(target):
        if i == 'L': target_L.append(idx)
        if i == 'R': target_R.append(idx)
    for i, j in zip(start_L, target_L):
        if i < j: return False
    for i, j in zip(start_R, target_R):
        if i > j: return False
    return True
```



### 2338. Count the Number of Ideal Arrays

You are given two integers `n` and `maxValue`, which are used to describe an **ideal** array.

A **0-indexed** integer array `arr` of length `n` is considered **ideal** if the following conditions hold:

- Every `arr[i]` is a value from `1` to `maxValue`, for `0 <= i < n`.
- Every `arr[i]` is divisible by `arr[i - 1]`, for `0 < i < n`.

Return *the number of **distinct** ideal arrays of length* `n`. Since the answer may be very large, return it modulo `10 ** 9 + 7`.

 

**Example 1:**

```
Input: n = 2, maxValue = 5
Output: 10
Explanation: The following are the possible ideal arrays:
- Arrays starting with the value 1 (5 arrays): [1,1], [1,2], [1,3], [1,4], [1,5]
- Arrays starting with the value 2 (2 arrays): [2,2], [2,4]
- Arrays starting with the value 3 (1 array): [3,3]
- Arrays starting with the value 4 (1 array): [4,4]
- Arrays starting with the value 5 (1 array): [5,5]
There are a total of 5 + 2 + 1 + 1 + 1 = 10 distinct ideal arrays.
```

**Example 2:**

```
Input: n = 5, maxValue = 3
Output: 11
Explanation: The following are the possible ideal arrays:
- Arrays starting with the value 1 (9 arrays): 
   - With no other distinct values (1 array): [1,1,1,1,1] 
   - With 2nd distinct value 2 (4 arrays): [1,1,1,1,2], [1,1,1,2,2], [1,1,2,2,2], [1,2,2,2,2]
   - With 2nd distinct value 3 (4 arrays): [1,1,1,1,3], [1,1,1,3,3], [1,1,3,3,3], [1,3,3,3,3]
- Arrays starting with the value 2 (1 array): [2,2,2,2,2]
- Arrays starting with the value 3 (1 array): [3,3,3,3,3]
There are a total of 9 + 1 + 1 = 11 distinct ideal arrays.
```

 

**Constraints:**

- `2 <= n <= 10 ** 4`
- `1 <= maxValue <= 10 ** 4`



> both parameters are way too large
>
> we cannot do O(n^2) dp computing



> Fascinating and mindblowing approach:
>
> - Ideal arrays are non-decreasing array
> - an ideal array will follow this pattern: 1 or more consecutive i + 1 or more consecutive i * k + ..
> - Hence, the first thing we can do is find a combination that consists of a set of integers: `seq = [a, b, c, ..]` that has the following two features:
>   - each value is unique
>   - `seq[i]` is a multiplier of `seq[i - 1]`
>
> Suppose we have `seq = [1, 2, 6]` (len = k = 3), and `n = 5`. How many distinct arrays can we generate from this? (with each value appear at least once)
>
> - the first approach is count the occurrence of each value
>   - one shortcoming: it’s difficult to ensure that each value appear at least once
>
> - the second approach is count the index when we changed to a new value
>
>   - this one is easier. Since the last value `6` will keep appearing until the last index, we don’t need to care it.
>
>   - the only thing we need to care is the last occurrence of 1 & 2.
>
>   - candidates shall be `range(0, n - 1 = 4)`, and we need to pick up `k - 1 = 2` distinct numbers 
>
>   - for instance, we pick 1 & 3, then we can generate an array = [1, 1, 2, 2, 6, 6]
>
>     ​														   index  ↑ 	↑	 ↑
>
>     ​															 last 1  last 2  last 6
>
>   - Let me introduct a function: `math.comb(n, k)` Return the number of ways to choose *`k`* items from *`n`* items without ==repetition== and ==without order==.
>   - `math.comb(n - 1, k - 1)` is what we need for each (n, k) pair



```python
def idealArrays(self, n: int, maxValue: int) -> int:
    MOD = 10 ** 9 + 7

    @functools.cache
    def cal(k):
        return math.comb(n - 1, k - 1)

    @functools.cache
    def dfs(last, size):
        ret = 0
        if size > 0: ret += cal(size)
        mul = 2
        while last * mul <= maxValue:
            ret += dfs(last * mul, size + 1)
            mul += 1
        return ret

    return (dfs(1, 1) + dfs(1, 0)) % MOD
```

