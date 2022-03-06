Given a positive integer `k`, you need to find the **length** of the **smallest** positive integer `n` such that `n` is divisible by `k`, and `n` only contains the digit `1`.

Return *the **length** of* `n`. If there is no such `n`, return -1.

**Note:** `n` may not fit in a 64-bit signed integer.

 

**Example 1:**

```
Input: k = 1
Output: 1
Explanation: The smallest answer is n = 1, which has length 1.
```

**Example 2:**

```
Input: k = 2
Output: -1
Explanation: There is no such positive integer n divisible by 2.
```

**Example 3:**

```
Input: k = 3
Output: 3
Explanation: The smallest answer is n = 111, which has length 3.
```

 

**Constraints:**

- `1 <= k <= 10 ** 5`

#### Discussion Approach:

- math provement:

```
1: 		1 % k = 1 = a
11:		11 % k = (10 % k + 1 % k) % k = 
		((1 % k) * 10 + 1) % k = (a * 10 + 1) % k = b
111:	111 % k = (110 % k + 1 % k) % k = 
		((11 % k) * 10 + 1) % k = (b * 10 + 1) % k = c
...
```

i 1’s remainder can be derived from the previous remainder: (i - 1) 1’s

Thus, if we visit the same remainder during our iteration without meeting a zero remainder. We can assure that n will never be divisible by k.

```python
def smallestRepunitDivByK(self, k: int) -> int:
    """
    pos: 1, 3, 7, 9
    imp: 2, 4, 5, 6, 8, 0
    """
    if k % 10 in [2, 4, 5, 6, 8, 0]: return -1
    visited = set()
    amount = 0
    curr_mod = 0
    while True:
        amount += 1
        curr_mod = (curr_mod * 10 + 1) % k
        if curr_mod == 0: return amount
        if curr_mod in visited: return -1
        visited.add(curr_mod)
```

- time/space complexity: O(k)

Provement: As provided above, we are iterating remainder of divider k in our code. According to [Pigeonhole principle](https://en.wikipedia.org/wiki/Pigeonhole_principle), the remainder of k will lie in range(k). Hence, the iteration will be called at most k times before visiting a duplicate remainder. Set `visited` will have k elements at most as well, which is our space complexity.