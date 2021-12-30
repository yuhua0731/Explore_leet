Given `n` pairs of parentheses, write a function to *generate all combinations of well-formed parentheses*.

 

**Example 1:**

```
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
```

**Example 2:**

```
Input: n = 1
Output: ["()"]
```

 

**Constraints:**

- `1 <= n <= 8`

```python
def generateParenthesis(self, n: int) -> List[str]:
    # stack
    ans = list()
    def find_next(pre: int, curr: str, remain: int):
        if remain < 0 or pre < 0: return
        if len(curr) == n * 2: ans.append(curr)
        find_next(pre + 1, curr + '(', remain - 1) # '('
        find_next(pre - 1, curr + ')', remain) # ')'

    find_next(0, '', n)
    return ans
```

