Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return *its sum*.

A **subarray** is a **contiguous** part of an array.

 

**Example 1:**

```
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
```

**Example 2:**

```
Input: nums = [1]
Output: 1
```

**Example 3:**

```
Input: nums = [5,4,-1,7,8]
Output: 23
```

 

**Constraints:**

- `1 <= nums.length <= 105`
- `-104 <= nums[i] <= 104`

 

**Follow up:** If you have figured out the `O(n)` solution, try coding another solution using the **divide and conquer** approach, which is more subtle.

#### First approach:

traverse input list, record previous positive sum and negative sum.

```python
def maxSubArray(self, nums: List[int]) -> int:
    pre_value = pre_posi = pre_nega = 0
    ans = overall_max = -float('inf')
    for i in nums:
        if i < 0:
            if pre_value >= 0: ans = max(pre_posi, ans) # +++-
            pre_nega += i
        else:
            if pre_value < 0:
                # ---+
                pre_posi = max(pre_posi + pre_nega, 0)
                pre_nega = 0
            pre_posi += i
        pre_value = i
        overall_max = max(overall_max, i)
    ans = max(pre_posi, ans)
    if overall_max < 0 and ans == 0: ans = overall_max
    return ans
```

Runtime: 736 ms, faster than 71.15% of Python3 online submissions for Maximum Subarray.

Memory Usage: 28.1 MB, less than 90.36% of Python3 online submissions for Maximum Subarray.

#### Second approach:

==dynamic programming== comes!

we record current sum and previous minimum sum during our traversal

update ans by choose the greater one between (curr_sum - pre_min_sum) and ans itself

Constant extra space and O(n) time

```python
 def maxSubArray(self, nums: List[int]) -> int:   
    # dp
    ans = nums[0]
    pre_min_sum = curr_sum = 0
    for i in nums:
        curr_sum += i
        ans = max(ans, curr_sum - pre_min_sum)
        pre_min_sum = min(pre_min_sum, curr_sum)
    return ans
```

