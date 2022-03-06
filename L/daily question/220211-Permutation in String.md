Given two strings `s1` and `s2`, return `true` *if* `s2` *contains a permutation of* `s1`*, or* `false` *otherwise*.

In other words, return `true` if one of `s1`'s permutations is the substring of `s2`.

 

**Example 1:**

```
Input: s1 = "ab", s2 = "eidbaooo"
Output: true
Explanation: s2 contains one permutation of s1 ("ba").
```

**Example 2:**

```
Input: s1 = "ab", s2 = "eidboaoo"
Output: false
```

 

**Constraints:**

- `1 <= s1.length, s2.length <= 104`
- `s1` and `s2` consist of lowercase English letters.

```python
def checkInclusion(self, s1: str, s2: str) -> bool:
    """check if one of s1's permutation is the substring of s2

    Args:
        s1 (str): given string 1
        s2 (str): given string 2

    Returns:
        bool: return True if we found the permutation
    """
    cnt = collections.Counter(s1)
    start = end = 0
    print(cnt)
    while end < len(s2):
        if s2[end] not in cnt:
            start = end = end + 1
            cnt = collections.Counter(s1)
            continue
        cnt[s2[end]] -= 1
        while cnt[s2[end]] < 0:
            cnt[s2[start]] += 1
            start += 1
        if end - start + 1 == len(s1): return True
        end += 1 
    return False
```

