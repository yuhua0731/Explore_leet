Nearly everyone has used the [Multiplication Table](https://en.wikipedia.org/wiki/Multiplication_table). The multiplication table of size `m x n` is an integer matrix `mat` where `mat[i][j] == i * j` (**1-indexed**).

Given three integers `m`, `n`, and `k`, return *the* `kth` *smallest element in the* `m x n` *multiplication table*.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/05/02/multtable1-grid.jpg)

```
Input: m = 3, n = 3, k = 5
Output: 3
Explanation: The 5th smallest number is 3.
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2021/05/02/multtable2-grid.jpg)

```
Input: m = 2, n = 3, k = 6
Output: 6
Explanation: The 6th smallest number is 6.
```

 

**Constraints:**

- `1 <= m, n <= 3 * 104`
- `1 <= k <= m * n`



#### First Approach:

Form a **unique heapq**, in each loop:

Pop the smallest product from heapq, increment i/j by 1, push back new product.



Pesudo code:

```python
for _ in range(k):
    prod, i, j = pop out from heapq
    new_prod = prod_, i + 1, j
    new_prod = prod_, i, j + 1
    if new_prod has not been visited:
        push new_prod back to heapq
return prod
```

```python
def findKthNumber(self, m: int, n: int, k: int) -> int:
    curr = [(1, 1, 1)] # prod, i, j | prod = i * j
    visited = set()
    next = [[0, 1], [1, 0]]
    for _ in range(k):
        prod, i, j = heapq.heappop(curr)
        for ni, nj in next:
            i_, j_ = i + ni, j + nj
            temp = (i_ * j_, i_, j_)
            if i_ <= m and j_ <= n and temp not in visited:
                visited.add(temp)
                heapq.heappush(curr, temp)
    return prod
```

logic correct, but got TLE for input 9895, 28405, 100787757

### Discussion:

✔️ ***Solution - I (Binary Search)***



We could think of generating the table, then sorting all the elements and returning `k`th element. This approach will take `O(m*n*log(mn))`. However, the given constraints `1 <= m, n <= 3 * 104` means that this solution won't pass since `m * n >= 9 * 108` which is too big to pass. A better solution would be to use binary search.



We know that the possible range of elements in the multiplication table is `[1, m*n]` and our answer should lie somewhere in there. The **answer is the smallest number `x` such that there are `k` numbers less than or equal to `x`**. For eg. `m=2, n=3, k=3`, then we have the values `[1,2,2,3,4,6]`. 2 has `k` numbers less or equal to itself. Thus, 2 is answer in this case. If `k=5`, then we had `4` in the table which had `k=5` numbers <= 4.



**How do we know if a number `x` has `k` numbers less than or equal to it?**



We obviously can't check by generating the table since that would take `O(m*n)` time. But we can do better by looking at how numbers appear in the table. In `row i`, the numbers we get, are of the form - **`[i, 2*i, 3*i,...,k*i,...n*i]`**. Using this observation, we can iterate over each row and count number of elements-`N` in that row that are less than or equal to `x`. We have -

```
		N * i <= x
=>      N <= x / i
However, there are at most n numbers in a row. 
So, we must also ensure we dont overcount.
Thus => N = min(n, x / i)
```

🤔 **Now, how does Binary search come into this?**
We have a way to count how many numbers are less than or equal to `x` in the table (let's call this function **`count(x)`**). We can simply apply binary search over the range **`[1, m*n]`** and iteratively check if `mid` has atleast `k` numbers less than or equal to `mid`.

- If `count(mid) < k`, there are less than `k` numbers which are less than or equal to `mid` in the table. So `mid` or any integer lower than it can't be our answer. So eliminate search space lower than mid by doing `low = mid + 1`
- If `count(mid) >= k`, there are at least `k` numbers (maybe more) which are less than or equal to `mid` in the table. So, **`mid` is a possible valid solution**. But there can be a smaller number than `mid` as well which has `count(.) >= k`. So, we mark current `mid` as possible answer `ans` and check for lower range as well by doing `high = mid-1`.



Finally, we return `ans` which will be the required `k` smallest number in the table.

```python
class Solution:
    def findKthNumber(self, m, n, k):
        def count(x):
            return sum(min(x // i, n) for i in range(1, m + 1))
			
        L, R, mid, ans = 0, m * n, 0, 0
        while L <= R:
            mid = (L + R) >> 1
            if count(mid) < k:
                L = mid + 1
            else:
                R, ans = mid - 1, mid
        return ans
```

**Time Complexity :** **`O(m*log(m*n))`**

1. We can further reduce the time complexity to `O(min(m,n)*log(m*n))` as well with a small change.
2. We can reduce it even further to `O(min(m,n)*log(k))` using another small observation.

*Comment below if you can figure it out along with how these optimization work :)*



***Space Complexity :*** **`O(1)`**

------

**📝 Proof: How do we know `ans` will always exist in multiplication table?**



Every time, we are only checking if `count(mid) >= k` and running the process till `L` & `R` narrow down to each other. But how do we know `ans` always exists in table?



For eg. Consider a 3x5 table -

```
m = 3,  n = 5,  k = 9

1  2  3  4   5
2  4  6  8   10
3  6  9  12  15

=> [1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 8, 9, 10, 12, 15]
We have count(7) = count(6) = 9. But 7 doesn't exist in our table
```

Let's say `ans` doesn't exist in the table. Then, we must have some other number `ans'` smaller than `ans` which does exist in table and for which `count(ans') >= k`. (Why `ans'` must be smaller than `ans`? Because if `ans` doesn't exist in table and `count(ans)>=k`, then there must be smaller number for which `count(.)>=k` and we need to find smallest such number.)



But, in the above solution, `ans` is the smallest number for which we found `count(mid) >= k` and then marked `mid` as `ans`. We can surely say that it's smallest since we also ran further search over range `[L, mid-1]` and we didnt find any `count([L...mid-1]) >= k`.



From above two observation, **we must conclude `ans' = ans` which is our final answer and does exist in the table.**



