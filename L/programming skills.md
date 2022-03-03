### 191. Number of 1 Bits

Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the [Hamming weight](http://en.wikipedia.org/wiki/Hamming_weight)).

**Note:**

- Note that in some languages, such as Java, there is no unsigned integer type. In this case, the input will be given as a signed integer type. It should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
- In Java, the compiler represents the signed integers using [2's complement notation](https://en.wikipedia.org/wiki/Two's_complement). Therefore, in **Example 3**, the input represents the signed integer. `-3`.

 

**Example 1:**

```
Input: n = 00000000000000000000000000001011
Output: 3
Explanation: The input binary string 00000000000000000000000000001011 has a total of three '1' bits.
```

**Example 2:**

```
Input: n = 00000000000000000000000010000000
Output: 1
Explanation: The input binary string 00000000000000000000000010000000 has a total of one '1' bit.
```

**Example 3:**

```
Input: n = 11111111111111111111111111111101
Output: 31
Explanation: The input binary string 11111111111111111111111111111101 has a total of thirty one '1' bits.
```

 

**Constraints:**

- The input must be a **binary string** of length `32`.

 

**Follow up:** If this function is called many times, how would you optimize it?

```python
def hammingWeight(self, n: int) -> int:
    return bin(n).count('1')
```

### 1281. Subtract the Product and Sum of Digits of an Integer

Given an integer number `n`, return the difference between the product of its digits and the sum of its digits.

 

**Example 1:**

```
Input: n = 234
Output: 15 
Explanation: 
Product of digits = 2 * 3 * 4 = 24 
Sum of digits = 2 + 3 + 4 = 9 
Result = 24 - 9 = 15
```

**Example 2:**

```
Input: n = 4421
Output: 21
Explanation: 
Product of digits = 4 * 4 * 2 * 1 = 32 
Sum of digits = 4 + 4 + 2 + 1 = 11 
Result = 32 - 11 = 21
```

 

**Constraints:**

- `1 <= n <= 10^5`

```python
def subtractProductAndSum(self, n: int) -> int:
    prod, s = 1, 0
    while n:
        i = n % 10
        prod *= i
        s += i
        n //= 10
    return prod - s
```

### 976. Largest Perimeter Triangle

Given an integer array `nums`, return *the largest perimeter of a triangle with a non-zero area, formed from three of these lengths*. If it is impossible to form any triangle of a non-zero area, return `0`.

 

**Example 1:**

```
Input: nums = [2,1,2]
Output: 5
```

**Example 2:**

```
Input: nums = [1,2,1]
Output: 0
```

 

**Constraints:**

- `3 <= nums.length <= 10 ** 4`
- `1 <= nums[i] <= 10 ** 6`

```python
def largestPerimeter(self, nums: List[int]) -> int:
    nums = sorted(nums, reverse = True)
    three = deque(nums[:3])
    nums = deque(nums[3:])
    while three[0] >= three[1] + three[2]:
        three.popleft()
        if not nums: return 0
        three.append(nums.popleft())
    return sum(three)
```

### 1779. Find Nearest Point That Has the Same X or Y Coordinate

You are given two integers, `x` and `y`, which represent your current location on a Cartesian grid: `(x, y)`. You are also given an array `points` where each `points[i] = [ai, bi]` represents that a point exists at `(ai, bi)`. A point is **valid** if it shares the same x-coordinate or the same y-coordinate as your location.

Return *the index **(0-indexed)** of the **valid** point with the smallest **Manhattan distance** from your current location*. If there are multiple, return *the valid point with the **smallest** index*. If there are no valid points, return `-1`.

The **Manhattan distance** between two points `(x1, y1)` and `(x2, y2)` is `abs(x1 - x2) + abs(y1 - y2)`.

 

**Example 1:**

```
Input: x = 3, y = 4, points = [[1,2],[3,1],[2,4],[2,3],[4,4]]
Output: 2
Explanation: Of all the points, only [3,1], [2,4] and [4,4] are valid. Of the valid points, [2,4] and [4,4] have the smallest Manhattan distance from your current location, with a distance of 1. [2,4] has the smallest index, so return 2.
```

**Example 2:**

```
Input: x = 3, y = 4, points = [[3,4]]
Output: 0
Explanation: The answer is allowed to be on the same location as your current location.
```

**Example 3:**

```
Input: x = 3, y = 4, points = [[2,3]]
Output: -1
Explanation: There are no valid points.
```

 

**Constraints:**

- `1 <= points.length <= 10 ** 4`
- `points[i].length == 2`
- `1 <= x, y, ai, bi <= 10 ** 4`

```python
def nearestValidPoint(self, x: int, y: int, points: List[List[int]]) -> int:
    ans = -1
    distance = float('inf')
    for idx, (i, j) in enumerate(points):
        if (i == x or y == j) and abs(x - i) + abs(y - j) < distance:
            distance = abs(x - i) + abs(y - j)
            ans = idx
    return ans
```





