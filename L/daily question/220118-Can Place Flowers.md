You have a long flowerbed in which some of the plots are planted, and some are not. However, flowers cannot be planted in **adjacent** plots.

Given an integer array `flowerbed` containing `0`'s and `1`'s, where `0` means empty and `1` means not empty, and an integer `n`, return *if* `n` new flowers can be planted in the `flowerbed` without violating the no-adjacent-flowers rule.

 

**Example 1:**

```
Input: flowerbed = [1,0,0,0,1], n = 1
Output: true
```

**Example 2:**

```
Input: flowerbed = [1,0,0,0,1], n = 2
Output: false
```

 

**Constraints:**

- `1 <= flowerbed.length <= 2 * 104`
- `flowerbed[i]` is `0` or `1`.
- There are no two adjacent flowers in `flowerbed`.
- `0 <= n <= flowerbed.length`

#### My approach

- recursion? better not
- Iterate input list, check how many adjacent 0 between two 1, let say i
- if i < 3, then no flower can be plant here
- if i >= 3, (i - 1) // 2, actually this equation is compatible for the situation when i < 3

```python
def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
    flowerbed = [1, 0] + flowerbed + [0, 1]
    amount = 0
    for i in flowerbed:
        if i == 1:
            if amount > 0:
                n -= (amount - 1) // 2
                amount = 0
            if n <= 0: return True
        else:
            amount += 1
            if (amount - 1) // 2 >= n: return True
    return False
```

