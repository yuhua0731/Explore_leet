### 2363. Merge Similar Items

You are given two 2D integer arrays, `items1` and `items2`, representing two sets of items. Each array `items` has the following properties:

- `items[i] = [valuei, weighti]` where `valuei` represents the **value** and `weighti` represents the **weight** of the `ith` item.
- The value of each item in `items` is **unique**.

Return *a 2D integer array* `ret` *where* `ret[i] = [valuei, weighti]`*,* *with* `weighti` *being the **sum of weights** of all items with value* `valuei`.

**Note:** `ret` should be returned in **ascending** order by value.

 

**Example 1:**

```
Input: items1 = [[1,1],[4,5],[3,8]], items2 = [[3,1],[1,5]]
Output: [[1,6],[3,9],[4,5]]
Explanation: 
The item with value = 1 occurs in items1 with weight = 1 and in items2 with weight = 5, total weight = 1 + 5 = 6.
The item with value = 3 occurs in items1 with weight = 8 and in items2 with weight = 1, total weight = 8 + 1 = 9.
The item with value = 4 occurs in items1 with weight = 5, total weight = 5.  
Therefore, we return [[1,6],[3,9],[4,5]].
```

**Example 2:**

```
Input: items1 = [[1,1],[3,2],[2,3]], items2 = [[2,1],[3,2],[1,3]]
Output: [[1,4],[2,4],[3,4]]
Explanation: 
The item with value = 1 occurs in items1 with weight = 1 and in items2 with weight = 3, total weight = 1 + 3 = 4.
The item with value = 2 occurs in items1 with weight = 3 and in items2 with weight = 1, total weight = 3 + 1 = 4.
The item with value = 3 occurs in items1 with weight = 2 and in items2 with weight = 2, total weight = 2 + 2 = 4.
Therefore, we return [[1,4],[2,4],[3,4]].
```

**Example 3:**

```
Input: items1 = [[1,3],[2,2]], items2 = [[7,1],[2,2],[1,4]]
Output: [[1,7],[2,4],[7,1]]
Explanation:
The item with value = 1 occurs in items1 with weight = 3 and in items2 with weight = 4, total weight = 3 + 4 = 7. 
The item with value = 2 occurs in items1 with weight = 2 and in items2 with weight = 2, total weight = 2 + 2 = 4. 
The item with value = 7 occurs in items2 with weight = 1, total weight = 1.
Therefore, we return [[1,7],[2,4],[7,1]].
```

 

**Constraints:**

- `1 <= items1.length, items2.length <= 1000`
- `items1[i].length == items2[i].length == 2`
- `1 <= valuei, weighti <= 1000`
- Each `valuei` in `items1` is **unique**.
- Each `valuei` in `items2` is **unique**.

```python
def mergeSimilarItems(self, items1: List[List[int]], items2: List[List[int]]) -> List[List[int]]:
    ret = dict()
    for i, j in items1 + items2:
        if i not in ret: ret[i] = 0
        ret[i] += j
    return sorted([[i, j] for i, j in ret.items()])
```



### 2364. Count Number of Bad Pairs

You are given a **0-indexed** integer array `nums`. A pair of indices `(i, j)` is a **bad pair** if `i < j` and `j - i != nums[j] - nums[i]`.

Return *the total number of **bad pairs** in* `nums`.

 

**Example 1:**

```
Input: nums = [4,1,3,3]
Output: 5
Explanation: The pair (0, 1) is a bad pair since 1 - 0 != 1 - 4.
The pair (0, 2) is a bad pair since 2 - 0 != 3 - 4, 2 != -1.
The pair (0, 3) is a bad pair since 3 - 0 != 3 - 4, 3 != -1.
The pair (1, 2) is a bad pair since 2 - 1 != 3 - 1, 1 != 2.
The pair (2, 3) is a bad pair since 3 - 2 != 3 - 3, 1 != 0.
There are a total of 5 bad pairs, so we return 5.
```

**Example 2:**

```
Input: nums = [1,2,3,4,5]
Output: 0
Explanation: There are no bad pairs.
```

 

**Constraints:**

- `1 <= nums.length <= 10 ** 5`
- `1 <= nums[i] <= 10 ** 9`

> two tricks:
>
> - Instead of counting the bad pairs, it is easier to count the good pairs
> - look at the equation: j - i == nums[j] - nums[i]
>   - if we maintain the previous numbers as nums[j] - nums[i] + i and checking if j == previous records, we need to update all records every loop, which will definitely explode our time complexity
>   - let’s convert the equation to: j - nums[j] == i - nums[i]. Then previous numbers become independent to j. Please read the code for the magic.

```python
def countBadPairs(self, nums: List[int]) -> int:
    # brute force
    # we maintain a dict: key = expect value, value = count
    # convert j - i = nums[j] - nums[i] to j - nums[j] = i - nums[i]
    ex = collections.defaultdict(int)
    ret = 0
    for idx, i in enumerate(nums):
        exp = idx - i
        ret += ex[exp]
        ex[exp] += 1
    n = len(nums)
    return n * (n - 1) // 2 - ret
```



### 2365. Task Scheduler II

You are given a **0-indexed** array of positive integers `tasks`, representing tasks that need to be completed **in order**, where `tasks[i]` represents the **type** of the `ith` task.

You are also given a positive integer `space`, which represents the **minimum** number of days that must pass **after** the completion of a task before another task of the **same** type can be performed.

Each day, until all tasks have been completed, you must either:

- Complete the next task from `tasks`, or
- Take a break.

Return *the **minimum** number of days needed to complete all tasks*.

 

**Example 1:**

```
Input: tasks = [1,2,1,2,3,1], space = 3
Output: 9
Explanation:
One way to complete all tasks in 9 days is as follows:
Day 1: Complete the 0th task.
Day 2: Complete the 1st task.
Day 3: Take a break.
Day 4: Take a break.
Day 5: Complete the 2nd task.
Day 6: Complete the 3rd task.
Day 7: Take a break.
Day 8: Complete the 4th task.
Day 9: Complete the 5th task.
It can be shown that the tasks cannot be completed in less than 9 days.
```

**Example 2:**

```
Input: tasks = [5,8,8,5], space = 2
Output: 6
Explanation:
One way to complete all tasks in 6 days is as follows:
Day 1: Complete the 0th task.
Day 2: Complete the 1st task.
Day 3: Take a break.
Day 4: Take a break.
Day 5: Complete the 2nd task.
Day 6: Complete the 3rd task.
It can be shown that the tasks cannot be completed in less than 6 days.
```

 

**Constraints:**

- `1 <= tasks.length <= 10 ** 5`
- `1 <= tasks[i] <= 10 ** 9`
- `1 <= space <= tasks.length`

```python
def taskSchedulerII(self, tasks: List[int], space: int) -> int:
    pre = dict()
    day = 0
    for i in tasks:
        day += 1
        if i not in pre: pre[i] = day
        else:
            day = max(day, pre[i] + space + 1)
            pre[i] = day
    return day
```



### 2366. Minimum Replacements to Sort the Array

You are given a **0-indexed** integer array `nums`. In one operation you can replace any element of the array with **any two** elements that **sum** to it.

- For example, consider `nums = [5,6,7]`. In one operation, we can replace `nums[1]` with `2` and `4` and convert `nums` to `[5,2,4,7]`.

Return *the minimum number of operations to make an array that is sorted in **non-decreasing** order*.

 

**Example 1:**

```
Input: nums = [3,9,3]
Output: 2
Explanation: Here are the steps to sort the array in non-decreasing order:
- From [3,9,3], replace the 9 with 3 and 6 so the array becomes [3,3,6,3]
- From [3,3,6,3], replace the 6 with 3 and 3 so the array becomes [3,3,3,3,3]
There are 2 steps to sort the array in non-decreasing order. Therefore, we return 2.
```

**Example 2:**

```
Input: nums = [1,2,3,4,5]
Output: 0
Explanation: The array is already in non-decreasing order. Therefore, we return 0. 
```

 

**Constraints:**

- `1 <= nums.length <= 10 ** 5`
- `1 <= nums[i] <= 10 ** 9`

> 🫠 Sorry my brain got no idea..
>
> One hint: since we can only make numbers smaller, try to traverse from right to left



> Follow-up challenge: 
>
> 
>
> When you traverse to a number `N`, and you find out that this number should be splitted into `k` numbers less than or equal to a constant `pre`.
>
> 
>
> We have three requirements:
>
> - all k numbers should be less than or equal to pre
> - make k as small as possible
> - the minimal number among k number should be as big as possible
>
> This fact is important to us when splitting: **for any positive integer n and all 1 <= k <= n, it's possible to represent n as the sum of k numbers, such that all numbers differ by no more than 1.**
>
> 
>
> ==Demonstration==: 
>
> let’s use q, r to represent for the quotient and remainder from `N / pre`
>
> - if r == 0, then N can be splitted into q numbers, and they are all equal to pre.
> - if r > 0, then N should be splitted into at least q + 1 numbers
>   - the minimal number is N // (q + 1)
>
> In this way, we can minimize the k, and maximize the minimal element as well.

```python
def minimumReplacement(self, nums: List[int]) -> int:
    right = nums[-1]
    ret = 0
    for i in reversed(nums[:len(nums) - 1]):
        if i <= right: 
            right = i
            continue
        q, r = i // right, i % right
        if r == 0: ret += q - 1
        else: 
            ret += q
            right = (i // (q + 1))
    return ret
```

