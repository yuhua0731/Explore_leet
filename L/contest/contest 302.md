### 2341. Maximum Number of Pairs in Array

You are given a **0-indexed** integer array `nums`. In one operation, you may do the following:

- Choose **two** integers in `nums` that are **equal**.
- Remove both integers from `nums`, forming a **pair**.

The operation is done on `nums` as many times as possible.

Return *a **0-indexed** integer array* `answer` *of size* `2` *where* `answer[0]` *is the number of pairs that are formed and* `answer[1]` *is the number of leftover integers in* `nums` *after doing the operation as many times as possible*.

 

**Example 1:**

```
Input: nums = [1,3,2,1,3,2,2]
Output: [3,1]
Explanation:
Form a pair with nums[0] and nums[3] and remove them from nums. Now, nums = [3,2,3,2,2].
Form a pair with nums[0] and nums[2] and remove them from nums. Now, nums = [2,2,2].
Form a pair with nums[0] and nums[1] and remove them from nums. Now, nums = [2].
No more pairs can be formed. A total of 3 pairs have been formed, and there is 1 number leftover in nums.
```

**Example 2:**

```
Input: nums = [1,1]
Output: [1,0]
Explanation: Form a pair with nums[0] and nums[1] and remove them from nums. Now, nums = [].
No more pairs can be formed. A total of 1 pair has been formed, and there are 0 numbers leftover in nums.
```

**Example 3:**

```
Input: nums = [0]
Output: [0,1]
Explanation: No pairs can be formed, and there is 1 number leftover in nums.
```

 

**Constraints:**

- `1 <= nums.length <= 100`
- `0 <= nums[i] <= 100`

```python
def numberOfPairs(self, nums: List[int]) -> List[int]:
    cnt = collections.Counter(nums)
    return [sum([i >> 1 for i in cnt.values()]), sum([i & 1 for i in cnt.values()])]
```



### 2342. Max Sum of a Pair With Equal Sum of Digits

You are given a **0-indexed** array `nums` consisting of **positive** integers. You can choose two indices `i` and `j`, such that `i != j`, and the sum of digits of the number `nums[i]` is equal to that of `nums[j]`.

Return *the **maximum** value of* `nums[i] + nums[j]` *that you can obtain over all possible indices* `i` *and* `j` *that satisfy the conditions.*

 

**Example 1:**

```
Input: nums = [18,43,36,13,7]
Output: 54
Explanation: The pairs (i, j) that satisfy the conditions are:
- (0, 2), both numbers have a sum of digits equal to 9, and their sum is 18 + 36 = 54.
- (1, 4), both numbers have a sum of digits equal to 7, and their sum is 43 + 7 = 50.
So the maximum sum that we can obtain is 54.
```

**Example 2:**

```
Input: nums = [10,12,19,14]
Output: -1
Explanation: There are no two numbers that satisfy the conditions, so we return -1.
```

 

**Constraints:**

- `1 <= nums.length <= 10 ** 5`
- `1 <= nums[i] <= 10 ** 9`

```python
def maximumSum(self, nums: List[int]) -> int:
    ret = collections.defaultdict(list)
    for i in nums:
        tmp = sum(int(j) for j in str(i))
        ret[tmp].append(i)
    return max([-1] + [sum(sorted(i, reverse = True)[:2]) for i in ret.values() if len(i) >= 2])
```



### 2343. Query Kth Smallest Trimmed Number

You are given a **0-indexed** array of strings `nums`, where each string is of **equal length** and consists of only digits.

You are also given a **0-indexed** 2D integer array `queries` where `queries[i] = [ki, trimi]`. For each `queries[i]`, you need to:

- **Trim** each number in `nums` to its **rightmost** `trimi` digits.
- Determine the **index** of the `kith` smallest trimmed number in `nums`. If two trimmed numbers are equal, the number with the **lower** index is considered to be smaller.
- Reset each number in `nums` to its original length.

Return *an array* `answer` *of the same length as* `queries`, *where* `answer[i]` *is the answer to the* `ith` *query.*

**Note**:

- To trim to the rightmost `x` digits means to keep removing the leftmost digit, until only `x` digits remain.
- Strings in `nums` may contain leading zeros.

 

**Example 1:**

```
Input: nums = ["102","473","251","814"], queries = [[1,1],[2,3],[4,2],[1,2]]
Output: [2,2,1,0]
Explanation:
1. After trimming to the last digit, nums = ["2","3","1","4"]. The smallest number is 1 at index 2.
2. Trimmed to the last 3 digits, nums is unchanged. The 2nd smallest number is 251 at index 2.
3. Trimmed to the last 2 digits, nums = ["02","73","51","14"]. The 4th smallest number is 73.
4. Trimmed to the last 2 digits, the smallest number is 2 at index 0.
   Note that the trimmed number "02" is evaluated as 2.
```

**Example 2:**

```
Input: nums = ["24","37","96","04"], queries = [[2,1],[2,2]]
Output: [3,0]
Explanation:
1. Trimmed to the last digit, nums = ["4","7","6","4"]. The 2nd smallest number is 4 at index 3.
   There are two occurrences of 4, but the one at index 0 is considered smaller than the one at index 3.
2. Trimmed to the last 2 digits, nums is unchanged. The 2nd smallest number is 24.
```

 

**Constraints:**

- `1 <= nums.length <= 100`
- `1 <= nums[i].length <= 100`
- `nums[i]` consists of only digits.
- All `nums[i].length` are **equal**.
- `1 <= queries.length <= 100`
- `queries[i].length == 2`
- `1 <= ki <= nums.length`
- `1 <= trimi <= nums[i].length`

 

**Follow up:** Could you use the **Radix Sort Algorithm** to solve this problem? What will be the complexity of that solution?

> TODO: what is the Radix Sort Algorithm? 🫠🫠

```python
def smallestTrimmedNumbers(self, nums: List[str], queries: List[List[int]]) -> List[int]:
    # brute force first with memorization
    n = len(nums[0])

    @functools.cache
    def helper(digit):
        tmp = [int(i[n - digit:]) for i in nums]
        tmp = [[i, idx] for idx, i in enumerate(tmp)]
        tmp = [idx for _, idx in sorted(tmp)]
        return tmp

    return [helper(j)[i - 1] for i, j in queries]
```



### 2344. Minimum Deletions to Make Array Divisible

You are given two positive integer arrays `nums` and `numsDivide`. You can delete any number of elements from `nums`.

Return *the **minimum** number of deletions such that the **smallest** element in* `nums` ***divides** all the elements of* `numsDivide`. If this is not possible, return `-1`.

Note that an integer `x` divides `y` if `y % x == 0`.

 

**Example 1:**

```
Input: nums = [2,3,2,4,3], numsDivide = [9,6,9,3,15]
Output: 2
Explanation: 
The smallest element in [2,3,2,4,3] is 2, which does not divide all the elements of numsDivide.
We use 2 deletions to delete the elements in nums that are equal to 2 which makes nums = [3,4,3].
The smallest element in [3,4,3] is 3, which divides all the elements of numsDivide.
It can be shown that 2 is the minimum number of deletions needed.
```

**Example 2:**

```
Input: nums = [4,3,6], numsDivide = [8,2,6,10]
Output: -1
Explanation: 
We want the smallest element in nums to divide all the elements of numsDivide.
There is no way to delete elements from nums to allow this.
```

 

**Constraints:**

- `1 <= nums.length, numsDivide.length <= 10 ** 5`
- `1 <= nums[i], numsDivide[i] <= 10 ** 9`

```python
def minOperations(self, nums: List[int], numsDivide: List[int]) -> int:
    # simple gcd function
    def gcd(a: int, b: int) -> int:
        if a < b: a, b = b, a
        while b != 0:
            a %= b
            a, b = b, a
        return a

    g = numsDivide[0]
    for i in numsDivide:
        g = gcd(g, i)

    cnt = collections.Counter(nums)
    ret = 0
    for k in sorted(cnt.keys()):
        if g % k == 0: return ret
        else: ret += cnt[k]
    return -1
```

