### 2368. Reachable Nodes With Restrictions

There is an undirected tree with `n` nodes labeled from `0` to `n - 1` and `n - 1` edges.

You are given a 2D integer array `edges` of length `n - 1` where `edges[i] = [ai, bi]` indicates that there is an edge between nodes `ai` and `bi` in the tree. You are also given an integer array `restricted` which represents **restricted** nodes.

Return *the **maximum** number of nodes you can reach from node* `0` *without visiting a restricted node.*

Note that node `0` will **not** be a restricted node.

 

**Example 1:**

![img](image_backup/contest 305/ex1drawio.png)

```
Input: n = 7, edges = [[0,1],[1,2],[3,1],[4,0],[0,5],[5,6]], restricted = [4,5]
Output: 4
Explanation: The diagram above shows the tree.
We have that [0,1,2,3] are the only nodes that can be reached from node 0 without visiting a restricted node.
```

**Example 2:**

![img](image_backup/contest 305/ex2drawio.png)

```
Input: n = 7, edges = [[0,1],[0,2],[0,5],[0,4],[3,2],[6,5]], restricted = [4,2,1]
Output: 3
Explanation: The diagram above shows the tree.
We have that [0,5,6] are the only nodes that can be reached from node 0 without visiting a restricted node.
```

 

**Constraints:**

- `2 <= n <= 105`
- `edges.length == n - 1`
- `edges[i].length == 2`
- `0 <= ai, bi < n`
- `ai != bi`
- `edges` represents a valid tree.
- `1 <= restricted.length < n`
- `1 <= restricted[i] < n`
- All the values of `restricted` are **unique**.

> 🫠 To avoid ending up with TLE. We need to do some pre-process.
>
> Convert restricted from a list to a hashset. The time complexity of search an element in a list is O(n), while it is O(1) in a set.

```python
def reachableNodes(self, n: int, edges: List[List[int]], restricted: List[int]) -> int:
    # union-find, TLE as well
    # uf = UnionFind(n)
    # for i, j in edges:
    #     if i not in restricted and j in restricted:
    #         uf.union(i, j)
    # tar = uf.find(0)
    # return len([i for i in range(n) if uf.find(i) == tar])


    # TLE
    restricted = set(restricted) # ... turns out we will end up with TLE without converting restricted to a hashset
    graph = collections.defaultdict(list)

    for i, j in edges:
        graph[i].append(j)
        graph[j].append(i)

    curr = [0]
    visited = {0}

    while curr:
        nxt = []
        for i in curr:
            for n in graph[i]:
                if n not in visited and n not in restricted:
                    nxt.append(n)
                    visited.add(n)
        curr = nxt
    return len(visited)
```



### 2369. Check if There is a Valid Partition For The Array

You are given a **0-indexed** integer array `nums`. You have to partition the array into one or more **==contiguous==** subarrays.

We call a partition of the array **valid** if each of the obtained subarrays satisfies **one** of the following conditions:

1. The subarray consists of **exactly** `2` equal elements. For example, the subarray `[2,2]` is good.
2. The subarray consists of **exactly** `3` equal elements. For example, the subarray `[4,4,4]` is good.
3. The subarray consists of **exactly** `3` consecutive increasing elements, that is, the difference between adjacent elements is `1`. For example, the subarray `[3,4,5]` is good, but the subarray `[1,3,5]` is not.

Return `true` *if the array has **at least** one valid partition*. Otherwise, return `false`.

 

**Example 1:**

```
Input: nums = [4,4,4,5,6]
Output: true
Explanation: The array can be partitioned into the subarrays [4,4] and [4,5,6].
This partition is valid, so we return true.
```

**Example 2:**

```
Input: nums = [1,1,1,2]
Output: false
Explanation: There is no valid partition for this array.
```

 

**Constraints:**

- `2 <= nums.length <= 10 ** 5`
- `1 <= nums[i] <= 10 ** 6`

```python
def validPartition(self, nums: List[int]) -> bool:
    # contiguous subarrays
    # dp, dp[i] = if we can partition nums[:i]
    dp = [True] # initialize with no element

    for i in range(len(nums)):
        tmp = False
        # end with 3 consecutive increasing elements
        if i >= 2 and nums[i] == nums[i - 1] + 1 and nums[i - 1] == nums[i - 2] + 1:
            tmp = tmp or dp[-3]
        if i >= 1 and nums[i] == nums[i - 1]:
            tmp = tmp or dp[-2]
        if i >= 2 and nums[i] == nums[i - 1] == nums[i - 2]:
            tmp = tmp or dp[-3]
        dp.append(tmp)
    return dp[-1]
```



### 2370. Longest Ideal Subsequence

You are given a string `s` consisting of lowercase letters and an integer `k`. We call a string `t` **ideal** if the following conditions are satisfied:

- `t` is a **subsequence** of the string `s`.
- The absolute difference in the alphabet order of every two **adjacent** letters in `t` is less than or equal to `k`.

Return *the length of the **longest** ideal string*.

A **subsequence** is a string that can be derived from another string by deleting some or no characters without changing the order of the remaining characters.

**Note** that the alphabet order is not cyclic. For example, the absolute difference in the alphabet order of `'a'` and `'z'` is `25`, not `1`.

 

**Example 1:**

```
Input: s = "acfgbd", k = 2
Output: 4
Explanation: The longest ideal string is "acbd". The length of this string is 4, so 4 is returned.
Note that "acfgbd" is not ideal because 'c' and 'f' have a difference of 3 in alphabet order.
```

**Example 2:**

```
Input: s = "abcd", k = 3
Output: 4
Explanation: The longest ideal string is "abcd". The length of this string is 4, so 4 is returned.
```

 

**Constraints:**

- `1 <= s.length <= 10 ** 5`
- `0 <= k <= 25`
- `s` consists of lowercase English letters.

> O(n) time and O(26) space

```python
def longestIdealString(self, s: str, k: int) -> int:
    """a better approach

    we maintain a list of length 26 called letter
    letter[i] presents the longest length of ideal subsequences that end with letter 'a' + i
    """
    if k == 25: return len(s)
    letter = [0] * 26
    for i in s:
        idx = ord(i) - ord('a')
        left_boundary, right_boundary = max(0, idx - k), min(25, idx + k)
        letter[idx] = max(letter[left_boundary : right_boundary + 1]) + 1
    return max(letter)
```

