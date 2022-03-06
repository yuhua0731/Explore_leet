Given string num representing a non-negative integer `num`, and an integer `k`, return *the smallest possible integer after removing* `k` *digits from* `num`.

 

**Example 1:**

```
Input: num = "1432219", k = 3
Output: "1219"
Explanation: Remove the three digits 4, 3, and 2 to form the new number 1219 which is the smallest.
```

**Example 2:**

```
Input: num = "10200", k = 1
Output: "200"
Explanation: Remove the leading 1 and the number is 200. Note that the output must not contain leading zeroes.
```

**Example 3:**

```
Input: num = "10", k = 2
Output: "0"
Explanation: Remove all the digits from the number and it is left with nothing which is 0.
```

 

**Constraints:**

- `1 <= k <= num.length <= 105`
- `num` consists of only digits.
- `num` does not have any leading zeros except for the zero itself.

> removing k digits from num
>
> total n digits
>
> there will be n - k digits left in our result

#### First approach

```python
def removeKdigits(self, num: str, k: int) -> str:
    # recursion
    n = len(num)
    def pick_one(idx: int, remain: int):
        """pick the smallest leading digit for remaining digits

        Args:
            idx (int): the index to search start
            remain (int): remaining digits we should pick
        """
        if remain == 0: return ''
        # n - idx > remain is guaranteed
        # pick the smallest leading digit

        # remaining num = num[idx:]
        # remain digits to pick is remain
        candidates = num[idx : n - (remain - 1)]
        digit, nxt_idx = sorted([[v, i] for i, v in enumerate(candidates)])[0]
        return digit + pick_one(idx + nxt_idx + 1, remain - 1)
    ret = pick_one(0, n - k)
    while len(ret) > 1 and ret[0] == '0':
        ret = ret[1:]
    return ret if ret else '0'
```

> Recursion: easy to implement, but has a terrible time complexity

#### Second approach

```python
def removeKdigits(self, num: str, k: int) -> str:
        """
             |<--               n               -->|
        num: '-------------------------------------'
             |<--   k   -->|<--      n - k      -->|
                ↑   ↑   ↑
             push these digits 
             into stack first 
             as candidates
        """
        # use stack, do not use recursion, since it cost too much time and space
        n = len(num)
        stack = []
        for i, d in enumerate(num[:k]):
            heapq.heappush(stack, (d, i))
        
        ret = ''
        pre_idx = -1
        while len(ret) < n - k:
            heapq.heappush(stack, (num[k + len(ret)], k + len(ret)))
            while stack[0][1] <= pre_idx:
                heapq.heappop(stack)
            d, i = heapq.heappop(stack)
            ret += d
            pre_idx = i
        while len(ret) > 1 and ret[0] == '0':
            ret = ret[1:]
        return ret if ret else '0'
```

#### discussion approach

```python
def removeKdigits(num: str, k: int) -> str:
    numStack = []
    
    # Construct a monotone increasing sequence of digits
    for digit in num:
        while k and numStack and numStack[-1] > digit:
            numStack.pop()
            k -= 1
    
        numStack.append(digit)
        print(numStack)
    
    # - Trunk the remaining K digits at the end
    # - in the case k==0: return the entire list
    finalStack = numStack[:-k] if k else numStack
    
    # trip the leading zeros
    return "".join(finalStack).lstrip('0') or "0"
```

