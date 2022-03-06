Given a string `s`, partition `s` such that every substring of the partition is a **palindrome**. Return all possible palindrome partitioning of `s`.

A **palindrome** string is a string that reads the same backward as forward.

 

**Example 1:**

```
Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]
```

**Example 2:**

```
Input: s = "a"
Output: [["a"]]
```

 

**Constraints:**

- `1 <= s.length <= 16`
- `s` contains only lowercase English letters.

#### My approach:

DP & recursion

==@cache== is really helpful

```python
def partition(self, s: str) -> List[List[str]]:
    # dp[i][j] = True if s[i:j] is a palindrome
    n = len(s)
    dp = [[False] * (n + 1) for _ in range(n + 1)]
    for diff in range(n + 1):
        for i in range(n + 1 - diff):
            j = i + diff
            if diff <= 1: dp[i][j] = True
            elif s[i] == s[j - 1]: dp[i][j] = dp[i + 1][j - 1]
                
	@cache
    def find_part(start: int):
        ans = list()
        for i in range(start, n):
            if dp[start][i + 1]: # is palindrome
                if i + 1 < n: ans += [[s[start:i + 1]] + p for p in find_part(i + 1)]
                else: ans.append([s[start:i + 1]])
        return ans
    return find_part(0)
```

