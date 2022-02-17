### 1710. Maximum Units on a Truck

You are assigned to put some amount of boxes onto **one truck**. You are given a 2D array `boxTypes`, where `boxTypes[i] = [numberOfBoxesi, numberOfUnitsPerBoxi]`:

- `numberOfBoxesi` is the number of boxes of type `i`.
- `numberOfUnitsPerBoxi` is the number of units in each box of the type `i`.

You are also given an integer `truckSize`, which is the **maximum** number of **boxes** that can be put on the truck. You can choose any boxes to put on the truck as long as the number of boxes does not exceed `truckSize`.

Return *the **maximum** total number of **units** that can be put on the truck.*

 

**Example 1:**

```
Input: boxTypes = [[1,3],[2,2],[3,1]], truckSize = 4
Output: 8
Explanation: There are:
- 1 box of the first type that contains 3 units.
- 2 boxes of the second type that contain 2 units each.
- 3 boxes of the third type that contain 1 unit each.
You can take all the boxes of the first and second types, and one box of the third type.
The total number of units will be = (1 * 3) + (2 * 2) + (1 * 1) = 8.
```

**Example 2:**

```
Input: boxTypes = [[5,10],[2,5],[4,7],[3,9]], truckSize = 10
Output: 91
```

 

**Constraints:**

- `1 <= boxTypes.length <= 1000`
- `1 <= numberOfBoxesi, numberOfUnitsPerBoxi <= 1000`
- `1 <= truckSize <= 106`

```python
def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
    boxTypes = [(-j, i) for i, j in boxTypes]
    heapq.heapify(boxTypes)
    ans = 0
    while truckSize and boxTypes:
        unit, cnt = heapq.heappop(boxTypes)
        cnt = min(cnt, truckSize)
        ans += -unit * cnt
        truckSize -= cnt
    return ans
```

### 1711. Count Good Meals

A **good meal** is a meal that contains **exactly two different food items** with a sum of deliciousness equal to a power of two.

You can pick **any** two different foods to make a good meal.

Given an array of integers `deliciousness` where `deliciousness[i]` is the deliciousness of the `ith` item of food, return *the number of different **good meals** you can make from this list modulo* `10 ** 9 + 7`.

Note that items with different indices are considered different even if they have the same deliciousness value.

 

**Example 1:**

```
Input: deliciousness = [1,3,5,7,9]
Output: 4
Explanation: The good meals are (1,3), (1,7), (3,5) and, (7,9).
Their respective sums are 4, 8, 8, and 16, all of which are powers of 2.
```

**Example 2:**

```
Input: deliciousness = [1,1,1,3,3,3,7]
Output: 15
Explanation: The good meals are (1,1) with 3 ways, (1,3) with 9 ways, and (1,7) with 3 ways.
```

 

**Constraints:**

- `1 <= deliciousness.length <= 10 ** 5`
- `0 <= deliciousness[i] <= 2 ** 20`

```python
def countPairs(self, deliciousness: List[int]) -> int:
    MOD = 10 ** 9 + 7
    cnt = collections.Counter(deliciousness)
    power = [2 ** i for i in range(22)]
    ans = 0
    for k, v in cnt.items():
        for p in power:
            if p - k in cnt:
                if k == p - k:
                    ans += cnt[k] * (cnt[k] - 1) // 2
                else:
                    ans += cnt[k] * cnt[p - k]
                ans %= MOD
        cnt[k] = 0
    return ans
```

### 1712. Ways to Split Array Into Three Subarrays

A split of an integer array is **good** if:

- The array is split into three **non-empty** contiguous subarrays - named `left`, `mid`, `right` respectively from left to right.
- The sum of the elements in `left` is less than or equal to the sum of the elements in `mid`, and the sum of the elements in `mid` is less than or equal to the sum of the elements in `right`.

Given `nums`, an array of **non-negative** integers, return *the number of **good** ways to split* `nums`. As the number may be too large, return it **modulo** `10 ** 9 + 7`.

 

**Example 1:**

```
Input: nums = [1,1,1]
Output: 1
Explanation: The only good way to split nums is [1] [1] [1].
```

**Example 2:**

```
Input: nums = [1,2,2,2,5,0]
Output: 3
Explanation: There are three good ways of splitting nums:
[1] [2] [2,2,5,0]
[1] [2,2] [2,5,0]
[1,2] [2,2] [5,0]
```

**Example 3:**

```
Input: nums = [3,2,1]
Output: 0
Explanation: There is no good way to split nums.
```

 

**Constraints:**

- `3 <= nums.length <= 10 ** 5`
- `0 <= nums[i] <= 10 ** 4`

```python
def waysToSplit(self, nums: List[int]) -> int:
    MOD = 10 ** 9 + 7

    for i in range(1, len(nums)):
        nums[i] += nums[i - 1]

    def find_low(idx):
        start, end = idx + 1, len(nums) - 1
        while start < end:
            mid = (start + end) >> 1
            if nums[mid] - 2 * nums[idx] >= 0:
                end = mid
            else:
                start = mid + 1
        return start

    def find_high(idx):
        start, end = idx + 1, len(nums) - 1
        while start < end:
            mid = (start + end) >> 1
            if nums[-1] - nums[mid] >= nums[mid] - nums[idx]:
                start = mid + 1
            else:
                end = mid
        return start

    # [:left + 1] [left + 1:right + 1] [right + 1:]
    ans = 0
    for left in range(len(nums)):
        if nums[-1] - nums[left] < 2 * nums[left]: break
        low, high = find_low(left), find_high(left)
        print(left, low, high)
        ans = (ans + max(0, high - low)) % MOD
    return ans
```

### 1713. Minimum Operations to Make a Subsequence

You are given an array `target` that consists of **distinct** integers and another integer array `arr` that **can** have duplicates.

In one operation, you can insert any integer at any position in `arr`. For example, if `arr = [1,4,1,2]`, you can add `3` in the middle and make it `[1,4,3,1,2]`. Note that you can insert the integer at the very beginning or end of the array.

Return *the **minimum** number of operations needed to make* `target` *a **subsequence** of* `arr`*.*

A **subsequence** of an array is a new array generated from the original array by deleting some elements (possibly none) without changing the remaining elements' relative order. For example, `[2,7,4]` is a subsequence of `[4,2,3,7,2,1,4]` (the underlined elements), while `[2,4,2]` is not.

 

**Example 1:**

```
Input: target = [5,1,3], arr = [9,4,2,3,4]
Output: 2
Explanation: You can add 5 and 1 in such a way that makes arr = [5,9,4,1,2,3,4], then target will be a subsequence of arr.
```

**Example 2:**

```
Input: target = [6,4,8,1,3,2], arr = [4,7,6,2,3,8,6,1]
Output: 3
```

 

**Constraints:**

- `1 <= target.length, arr.length <= 10 ** 5`
- `1 <= target[i], arr[i] <= 10 ** 9`
- `target` contains no duplicates.

> Longest common subsequence

I implemented a dp solution with a 2D dp array

time complexity = space complexity = O(m * n)

Sadly, this solution got TLE.



after viewing [Lee’s discussion post](https://leetcode.com/problems/minimum-operations-to-make-a-subsequence/discuss/999153/JavaC%2B%2BPython-LCS-to-LIS), I learned a better solution.

```python
def minOperations(self, target: List[int], arr: List[int]) -> int:
    h = {a: i for i, a in enumerate(target)}
    stack = []
    for a in arr:
        if a not in h: continue
        i = bisect.bisect_left(stack, h[a])
        if i == len(stack):
            stack.append(h[a])
        else:
            stack[i] = h[a]
    return len(target) - len(stack)
```

