Given an array `nums` of `n` integers where `nums[i]` is in the range `[1, n]`, return *an array of all the integers in the range* `[1, n]` *that do not appear in* `nums`.

 

**Example 1:**

```
Input: nums = [4,3,2,7,8,2,3,1]
Output: [5,6]
```

**Example 2:**

```
Input: nums = [1,1]
Output: [2]
```

 

**Constraints:**

- `n == nums.length`
- `1 <= n <= 105`
- `1 <= nums[i] <= n`

**Follow up:** Could you do it ==without extra space== and in `O(n)` runtime? You may assume the returned list does not count as extra space.

#### First approach:

Just using an additional variable to record the last matched integer.

traverse whole list

- if current element is greater than last_match, than update last_match and check if there is any missing number between last_match and current number.
- if current element is equal to last_match, do nothing.
- current element will never be less than last_match, cause we sort the whole list in advance.

```python
def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
    # without extra space and run in O(n) time
    nums.append(len(nums) + 1) 
    # append a corresponding number at the end of input list, to ensure that we will not miss the last check
    nums.sort()
    last_match = 0
    ans = list()
    for i in nums:
        if i > last_match: 
            for tmp in range(last_match + 1, i):
                ans.append(tmp)
            last_match = i
    return ans
```

Runtime: 372 ms, faster than 53.87% of Python3 online submissions for Find All Numbers Disappeared in an Array.

Memory Usage: 21.9 MB, less than 83.90% of Python3 online submissions for Find All Numbers Disappeared in an Array.

#### Second approach:

There is still some space that we can optimize our code to improve runtime.

one hint: in first approach, we record the last matched number, and once i is greater than this number, we always do missing number check, even though there is no missing number for some conditions that i == last_match + 1.



Hence, let’s replace last_match with next_match, if i == next_match, it indicates that there is no missing number, you just need to update next_match. 



This is just a tiny modification but it turns out our runtime improved by 10%.

```python
def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
    # without extra space and run in O(n) time
    nums.append(len(nums) + 1)
    nums.sort()
    next_match = 1
    ans = list()
    for i in nums:
        if i < next_match: continue
        if i > next_match: 
            for tmp in range(next_match, i):
                ans.append(tmp)
        next_match = i + 1
    return ans
```

Runtime: 344 ms, faster than 75.38% of Python3 online submissions for Find All Numbers Disappeared in an Array.

Memory Usage: 22 MB, less than 73.73% of Python3 online submissions for Find All Numbers Disappeared in an Array.

#### Discussion:

Even though my iteration has a time complexity of O(N), the sort function in advance will cost O(NlogN) time.

