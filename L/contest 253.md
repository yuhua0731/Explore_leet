### 1961. Check If String Is a Prefix of Array

Given a string `s` and an array of strings `words`, determine whether `s` is a **prefix string** of `words`.

A string `s` is a **prefix string** of `words` if `s` can be made by concatenating the first `k` strings in `words` for some **positive** `k` no larger than `words.length`.

Return `true` *if* `s` *is a **prefix string** of* `words`*, or* `false` *otherwise*.

 

**Example 1:**

```
Input: s = "iloveleetcode", words = ["i","love","leetcode","apples"]
Output: true
Explanation:
s can be made by concatenating "i", "love", and "leetcode" together.
```

**Example 2:**

```
Input: s = "iloveleetcode", words = ["apples","i","love","leetcode"]
Output: false
Explanation:
It is impossible to make s using a prefix of arr.
```

 

**Constraints:**

- `1 <= words.length <= 100`
- `1 <= words[i].length <= 20`
- `1 <= s.length <= 1000`
- `words[i]` and `s` consist of only lowercase English letters.

```python
def isPrefixString(self, s: str, words: List[str]) -> bool:
    for w in words:
        l = len(w)
        if l > len(s): break
        if w == s[:l]: 
            s = s[l:]
            if len(s) == 0: return True
        else: return False
    return False
```

### 1962. Remove Stones to Minimize the Total

You are given a **0-indexed** integer array `piles`, where `piles[i]` represents the number of stones in the `ith` pile, and an integer `k`. You should apply the following operation **exactly** `k` times:

- Choose any `piles[i]` and **remove** `floor(piles[i] / 2)` stones from it.

**Notice** that you can apply the operation on the **same** pile more than once.

Return *the **minimum** possible total number of stones remaining after applying the* `k` *operations*.

`floor(x)` is the **greatest** integer that is **smaller** than or **equal** to `x` (i.e., rounds `x` down).

 

**Example 1:**

```
Input: piles = [5,4,9], k = 2
Output: 12
Explanation: Steps of a possible scenario are:
- Apply the operation on pile 2. The resulting piles are [5,4,5].
- Apply the operation on pile 0. The resulting piles are [3,4,5].
The total number of stones in [3,4,5] is 12.
```

**Example 2:**

```
Input: piles = [4,3,6,7], k = 3
Output: 12
Explanation: Steps of a possible scenario are:
- Apply the operation on pile 2. The resulting piles are [4,3,3,7].
- Apply the operation on pile 3. The resulting piles are [4,3,3,4].
- Apply the operation on pile 0. The resulting piles are [2,3,3,4].
The total number of stones in [2,3,3,4] is 12.
```

 

**Constraints:**

- `1 <= piles.length <= 105`
- `1 <= piles[i] <= 104`
- `1 <= k <= 105`

```python
def minStoneSum(self, piles: List[int], k: int) -> int:
    total = sum(piles)
    piles = [-i for i in piles]
    heapq.heapify(piles)

    remove = 0
    for _ in range(k):
        curr = heapq.heappop(piles)
        remove += (-curr) // 2
        heapq.heappush(piles, curr + (-curr) // 2)
    return total - remove
```

### 1963. Minimum Number of Swaps to Make the String Balanced

You are given a **0-indexed** string `s` of **even** length `n`. The string consists of **exactly** `n / 2` opening brackets `'['` and `n / 2` closing brackets `']'`.

A string is called **balanced** if and only if:

- It is the empty string, or
- It can be written as `AB`, where both `A` and `B` are **balanced** strings, or
- It can be written as `[C]`, where `C` is a **balanced** string.

You may swap the brackets at **any** two indices **any** number of times.

Return *the **minimum** number of swaps to make* `s` ***balanced***.

 

**Example 1:**

```
Input: s = "][]["
Output: 1
Explanation: You can make the string balanced by swapping index 0 with index 3.
The resulting string is "[[]]".
```

**Example 2:**

```
Input: s = "]]][[["
Output: 2
Explanation: You can do the following to make the string balanced:
- Swap index 0 with index 4. s = "[]][][".
- Swap index 1 with index 5. s = "[[][]]".
The resulting string is "[[][]]".
```

**Example 3:**

```
Input: s = "[]"
Output: 0
Explanation: The string is already balanced.
```

 

**Constraints:**

- `n == s.length`
- `2 <= n <= 106`
- `n` is even.
- `s[i]` is either `'[' `or `']'`.
- The number of opening brackets `'['` equals `n / 2`, and the number of closing brackets `']'` equals `n / 2`.

```python
def minSwaps(self, s: str) -> int:
    mistake = 0
    remain = 0
    for i in s:
        if i == ']':
            if remain > 0:
                remain -= 1
            else:
                mistake += 1
        if i == '[':
            remain += 1
    mistake += remain
    return math.ceil(mistake / 4)
```

### 1964. Find the Longest Valid Obstacle Course at Each Position

You want to build some obstacle courses. You are given a **0-indexed** integer array `obstacles` of length `n`, where `obstacles[i]` describes the height of the `ith` obstacle.

For every index `i` between `0` and `n - 1` (**inclusive**), find the length of the **longest obstacle course** in `obstacles` such that:

- You choose any number of obstacles between `0` and `i` **inclusive**.
- You must include the `ith` obstacle in the course.
- You must put the chosen obstacles in the **same order** as they appear in `obstacles`.
- Every obstacle (except the first) is **taller** than or the **same height** as the obstacle immediately before it.

Return *an array* `ans` *of length* `n`, *where* `ans[i]` *is the length of the **longest obstacle course** for index* `i` *as described above*.

 

**Example 1:**

```
Input: obstacles = [1,2,3,2]
Output: [1,2,3,3]
Explanation: The longest valid obstacle course at each position is:
- i = 0: [1], [1] has length 1.
- i = 1: [1,2], [1,2] has length 2.
- i = 2: [1,2,3], [1,2,3] has length 3.
- i = 3: [1,2,3,2], [1,2,2] has length 3.
```

**Example 2:**

```
Input: obstacles = [2,2,1]
Output: [1,2,1]
Explanation: The longest valid obstacle course at each position is:
- i = 0: [2], [2] has length 1.
- i = 1: [2,2], [2,2] has length 2.
- i = 2: [2,2,1], [1] has length 1.
```

**Example 3:**

```
Input: obstacles = [3,1,5,6,4,2]
Output: [1,1,2,3,2,2]
Explanation: The longest valid obstacle course at each position is:
- i = 0: [3], [3] has length 1.
- i = 1: [3,1], [1] has length 1.
- i = 2: [3,1,5], [3,5] has length 2. [1,5] is also valid.
- i = 3: [3,1,5,6], [3,5,6] has length 3. [1,5,6] is also valid.
- i = 4: [3,1,5,6,4], [3,4] has length 2. [1,4] is also valid.
- i = 5: [3,1,5,6,4,2], [1,2] has length 2.
```

 

**Constraints:**

- `n == obstacles.length`
- `1 <= n <= 105`
- `1 <= obstacles[i] <= 107`

```python
def longestObstacleCourseAtEachPosition(self, obstacles: List[int]) -> List[int]:
    """find the longest obstacle course at each position

    Args:
        obstacles (List[int]): input obstacle list

    Returns:
        List[int]: a list of the longest length of non-decreasing list end with ith element
    """
    # bisect_left
    obs, ans = [], []
    for i in obstacles:
        idx = bisect.bisect_right(obs, i)
        ans.append(idx + 1)
        if idx == len(obs): obs.append(i)
        else: obs[idx] = i
    return ans
```

