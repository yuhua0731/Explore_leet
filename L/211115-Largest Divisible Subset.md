Given a set of **distinct** positive integers `nums`, return the largest subset `answer` such that every pair `(answer[i], answer[j])` of elements in this subset satisfies:

- `answer[i] % answer[j] == 0`, or
- `answer[j] % answer[i] == 0`

If there are multiple solutions, return any of them.

 

**Example 1:**

```
Input: nums = [1,2,3]
Output: [1,2]
Explanation: [1,3] is also accepted.
```

**Example 2:**

```
Input: nums = [1,2,4,8]
Output: [1,2,4,8]
```

 

**Constraints:**

- `1 <= nums.length <= 1000`
- `1 <= nums[i] <= 2 * 109`
- All the integers in `nums` are **unique**.

#### First approach:

1. 首先将输入列表递增排序，保证在遍历元素时所处理的数字保持递增。
2. 使用一个set记录目前为止找到的所有符合要求的子集，set中的元素为tuple（list类型无法被hash化）。由于使用的是递增遍历，当处理一个新数字时，只需要将子集中最大的元素取出，并判断是否能整除新数字，如果可以整除，则可将新数字添加到该子集中，并继续满足要求。
3. 最后取出set中元素个数最多的子集，将其转换为List并返回。

```python
def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
    sub = set()
    for n in sorted(nums):
        temp = set()
        temp.add((n,))
        for s in sub:
            if n % s[-1] == 0:
                temp.add((s[:] + (n,)))
        [sub.add(i) for i in temp]
    return sorted(list(sub), key=lambda x: len(x))[-1]
```

passed some tests, no logic error. However, got **TLE** for huge amount of input.

#### Second approach:

1. 在第一种方案里，并没有进行任何判断，直接将新元素作为一个新的tuple进行存放，导致后续会产生很多冗余逻辑。考虑以下情况：当存在两个子集[1,2,4]和[4]，对于任何不小于4的新数字i，如果可以被加进子集[4]中，则i必定可以被加进[1,2,4]中。所以，对于最大元素相同的多个子集，我们只需要保留size最大的即可。
2. 不再使用set类型，可以使用dict类型，key为子集中最大元素，value为list类型，存放的是元素个数最多的子集列表。list是mutable的，可以支持在循环时进行修改。

```python
def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
    sub = dict()
    for n in sorted(nums):
        temp = [n]
        for key, value in sub.items():
            if n % key == 0 and len(value) + 1 > len(temp):
                temp = value + [n] # form a new subset
        sub[n] = temp
    return sorted(sub.values(), key=lambda x: len(x))[-1]
```

passed as a submission. beats more than 70% in time.



