### d

191. Number of 1 Bits

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

### 1822. Sign of the Product of an Array

There is a function `signFunc(x)` that returns:

- `1` if `x` is positive.
- `-1` if `x` is negative.
- `0` if `x` is equal to `0`.

You are given an integer array `nums`. Let `product` be the product of all values in the array `nums`.

Return `signFunc(product)`.

 

**Example 1:**

```
Input: nums = [-1,-2,-3,-4,3,2,1]
Output: 1
Explanation: The product of all values in the array is 144, and signFunc(144) = 1
```

**Example 2:**

```
Input: nums = [1,5,0,2,-3]
Output: 0
Explanation: The product of all values in the array is 0, and signFunc(0) = 0
```

**Example 3:**

```
Input: nums = [-1,1,-1,1,-1]
Output: -1
Explanation: The product of all values in the array is -1, and signFunc(-1) = -1
```

 

**Constraints:**

- `1 <= nums.length <= 1000`
- `-100 <= nums[i] <= 100`

```python
def arraySign(self, nums: List[int]) -> int:
    cnt = 0
    for i in nums:
        if i == 0: return 0
        if i < 0: cnt += 1
    return -1 if cnt % 2 else 1
```

### 1502. Can Make Arithmetic Progression From Sequence

A sequence of numbers is called an **arithmetic progression** if the difference between any two consecutive elements is the same.

Given an array of numbers `arr`, return `true` *if the array can be rearranged to form an **arithmetic progression**. Otherwise, return* `false`.

 

**Example 1:**

```
Input: arr = [3,5,1]
Output: true
Explanation: We can reorder the elements as [1,3,5] or [5,3,1] with differences 2 and -2 respectively, between each consecutive elements.
```

**Example 2:**

```
Input: arr = [1,2,4]
Output: false
Explanation: There is no way to reorder the elements to obtain an arithmetic progression.
```

 

**Constraints:**

- `2 <= arr.length <= 1000`
- `-10 ** 6 <= arr[i] <= 10 ** 6`

```python
def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
    arr.sort()
    diff = arr[1] - arr[0]
    for i, j in zip(arr, arr[1:]):
        if j - i != diff: return False
    return True
```

### 202. Happy Number

Write an algorithm to determine if a number `n` is happy.

A **happy number** is a number defined by the following process:

- Starting with any positive integer, replace the number by the sum of the squares of its digits.
- Repeat the process until the number equals 1 (where it will stay), or it **loops endlessly in a cycle** which does not include 1.
- Those numbers for which this process **ends in 1** are happy.

Return `true` *if* `n` *is a happy number, and* `false` *if not*.

 

**Example 1:**

```
Input: n = 19
Output: true
Explanation:
1 ** 2 + 9 ** 2 = 82
8 ** 2 + 2 ** 2 = 68
6 ** 2 + 8 ** 2 = 100
1 ** 2 + 0 ** 2 + 02 = 1
```

**Example 2:**

```
Input: n = 2
Output: false
```

 

**Constraints:**

- `1 <= n <= 2 ** 31 - 1`

```python
def isHappy(self, n: int) -> bool:
    visited = set()
    digit = lambda x: int(x) ** 2
    while n != 1:
        n = sum(map(digit, list(str(n))))
        if n in visited: return False
        visited.add(n)
    return True
```

### 1790. Check if One String Swap Can Make Strings Equal

You are given two strings `s1` and `s2` of equal length. A **string swap** is an operation where you choose two indices in a string (not necessarily different) and swap the characters at these indices.

Return `true` *if it is possible to make both strings equal by performing **at most one string swap** on **exactly one** of the strings.* Otherwise, return `false`.

 

**Example 1:**

```
Input: s1 = "bank", s2 = "kanb"
Output: true
Explanation: For example, swap the first character with the last character of s2 to make "bank".
```

**Example 2:**

```
Input: s1 = "attack", s2 = "defend"
Output: false
Explanation: It is impossible to make them equal with one string swap.
```

**Example 3:**

```
Input: s1 = "kelb", s2 = "kelb"
Output: true
Explanation: The two strings are already equal, so no string swap operation is required.
```

 

**Constraints:**

- `1 <= s1.length, s2.length <= 100`
- `s1.length == s2.length`
- `s1` and `s2` consist of only lowercase English letters.

```python
def areAlmostEqual(self, s1: str, s2: str) -> bool:
    return len(s1) == len(s2) and sum(1 for i, j in zip(s1, s2) if i != j) <= 2 and sorted(s1) == sorted(s2)
```

### 589. N-ary Tree Preorder Traversal

Given the `root` of an n-ary tree, return *the preorder traversal of its nodes' values*.

Nary-Tree input serialization is represented in their level order traversal. Each group of children is separated by the null value (See examples)

 

**Example 1:**

![img](image_backup/programming skills/narytreeexample.png)

```
Input: root = [1,null,3,2,4,null,5,6]
Output: [1,3,5,6,2,4]
```

**Example 2:**

![img](image_backup/programming skills/sample_4_964.png)

```
Input: root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
Output: [1,2,3,6,7,11,14,4,8,12,5,9,13,10]
```

 

**Constraints:**

- The number of nodes in the tree is in the range `[0, 10 ** 4]`.
- `0 <= Node.val <= 10 ** 4`
- The height of the n-ary tree is less than or equal to `1000`.

 

**Follow up:** Recursive solution is trivial, could you do it iteratively?

```python
def preorder(self, root: 'Node') -> List[int]:
    curr = [root]
    ans = []
    while curr:
        temp = curr.pop()
        if not temp: return ans
        ans.append(temp.val)
        if temp.children: curr += temp.children[::-1]
    return ans
```

### 496. Next Greater Element I

The **next greater element** of some element `x` in an array is the **first greater** element that is **to the right** of `x` in the same array.

You are given two **distinct 0-indexed** integer arrays `nums1` and `nums2`, where `nums1` is a subset of `nums2`.

For each `0 <= i < nums1.length`, find the index `j` such that `nums1[i] == nums2[j]` and determine the **next greater element** of `nums2[j]` in `nums2`. If there is no next greater element, then the answer for this query is `-1`.

Return *an array* `ans` *of length* `nums1.length` *such that* `ans[i]` *is the **next greater element** as described above.*

 

**Example 1:**

```
Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
Output: [-1,3,-1]
Explanation: The next greater element for each value of nums1 is as follows:
- 4 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.
- 1 is underlined in nums2 = [1,3,4,2]. The next greater element is 3.
- 2 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.
```

**Example 2:**

```
Input: nums1 = [2,4], nums2 = [1,2,3,4]
Output: [3,-1]
Explanation: The next greater element for each value of nums1 is as follows:
- 2 is underlined in nums2 = [1,2,3,4]. The next greater element is 3.
- 4 is underlined in nums2 = [1,2,3,4]. There is no next greater element, so the answer is -1.
```

 

**Constraints:**

- `1 <= nums1.length <= nums2.length <= 1000`
- `0 <= nums1[i], nums2[i] <= 10 ** 4`
- All integers in `nums1` and `nums2` are **unique**.
- All the integers of `nums1` also appear in `nums2`.

 

**Follow up:** Could you find an `O(nums1.length + nums2.length)` solution?

```python
def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
    ans = dict()
    visited = []

    for x in nums2:
        # if the last previous element is less than x, we pop it out and set its answer to x
        # visited is guaranteed to be a non-increasing list
        while len(visited) and visited[-1] < x:
            ans[visited.pop()] = x
        visited.append(x)

    return [ans[i] if i in ans else -1 for i in nums1]
```

### 1232. Check If It Is a Straight Line

You are given an array `coordinates`, `coordinates[i] = [x, y]`, where `[x, y]` represents the coordinate of a point. Check if these points make a straight line in the XY plane.

 

 

**Example 1:**

![img](image_backup/programming skills/untitled-diagram-2.jpg)

```
Input: coordinates = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]
Output: true
```

**Example 2:**

**![img](image_backup/programming skills/untitled-diagram-1.jpg)**

```
Input: coordinates = [[1,1],[2,2],[3,4],[4,5],[5,6],[7,7]]
Output: false
```

 

**Constraints:**

- `2 <= coordinates.length <= 1000`
- `coordinates[i].length == 2`
- `-10^4 <= coordinates[i][0], coordinates[i][1] <= 10^4`
- `coordinates` contains no duplicate point.

```python
def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
    n = len(coordinates)
    x_coord = set(i for i, j in coordinates)
    if len(x_coord) == 1: return True
    if len(x_coord) < n: return False
    slope = set()
    for (i, j), (x, y) in zip(coordinates, coordinates[1:]):
        slope.add((y - j) / (x - i))
        if len(slope) > 1: return False
    return True
```

### 1309. Decrypt String from Alphabet to Integer Mapping

You are given a string `s` formed by digits and `'#'`. We want to map `s` to English lowercase characters as follows:

- Characters (`'a'` to `'i')` are represented by (`'1'` to `'9'`) respectively.
- Characters (`'j'` to `'z')` are represented by (`'10#'` to `'26#'`) respectively.

Return *the string formed after mapping*.

The test cases are generated so that a unique mapping will always exist.

 

**Example 1:**

```
Input: s = "10#11#12"
Output: "jkab"
Explanation: "j" -> "10#" , "k" -> "11#" , "a" -> "1" , "b" -> "2".
```

**Example 2:**

```
Input: s = "1326#"
Output: "acz"
```

 

**Constraints:**

- `1 <= s.length <= 1000`
- `s` consists of digits and the `'#'` letter.
- `s` will be a valid string such that mapping is always possible.

```python
def freqAlphabets(self, s: str) -> str:
    stack = 0
    ans = ''
    for i in s:
        if i == '#':
            if stack >= 100:
                single = stack // 100
                ans += ''.join(list(map(lambda x: chr(ord(x) - ord('1') + ord('a')), str(single))))
                stack %= 100
            ans += chr(int(stack) - 10 + ord('j'))
            stack = 0
        else:
            stack = stack * 10 + int(i)
    if stack:
        ans += ''.join(list(map(lambda x: chr(ord(x) - ord('1') + ord('a')), str(stack))))
    return ans
```

### 953. Verifying an Alien Dictionary

In an alien language, surprisingly, they also use English lowercase letters, but possibly in a different `order`. The `order` of the alphabet is some permutation of lowercase letters.

Given a sequence of `words` written in the alien language, and the `order` of the alphabet, return `true` if and only if the given `words` are sorted lexicographically in this alien language.

 

**Example 1:**

```
Input: words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"
Output: true
Explanation: As 'h' comes before 'l' in this language, then the sequence is sorted.
```

**Example 2:**

```
Input: words = ["word","world","row"], order = "worldabcefghijkmnpqstuvxyz"
Output: false
Explanation: As 'd' comes after 'l' in this language, then words[0] > words[1], hence the sequence is unsorted.
```

**Example 3:**

```
Input: words = ["apple","app"], order = "abcdefghijklmnopqrstuvwxyz"
Output: false
Explanation: The first three characters "app" match, and the second string is shorter (in size.) According to lexicographical rules "apple" > "app", because 'l' > '∅', where '∅' is defined as the blank character which is less than any other character (More info).
```

 

**Constraints:**

- `1 <= words.length <= 100`
- `1 <= words[i].length <= 20`
- `order.length == 26`
- All characters in `words[i]` and `order` are English lowercase letters.

```python
def isAlienSorted(self, words: List[str], order: str) -> bool:
    # generate a mapping dictionary from alien language to human language in earth
    alien = {i: earth for earth, i in zip([chr(a + ord('a')) for a in range(26)], order)}
    # translate all words to human readable language using the dictionary above
    words = list(map(lambda x: ''.join(alien[c] for c in x), words))
    # sort list to see if the original order is lexicographical
    return sorted(words) == words
```

### 1356. Sort Integers by The Number of 1 Bits

You are given an integer array `arr`. Sort the integers in the array in ascending order by the number of `1`'s in their binary representation and in case of two or more integers have the same number of `1`'s you have to sort them in ascending order.

Return *the array after sorting it*.

 

**Example 1:**

```
Input: arr = [0,1,2,3,4,5,6,7,8]
Output: [0,1,2,4,8,3,5,6,7]
Explantion: [0] is the only integer with 0 bits.
[1,2,4,8] all have 1 bit.
[3,5,6] have 2 bits.
[7] has 3 bits.
The sorted array by bits is [0,1,2,4,8,3,5,6,7]
```

**Example 2:**

```
Input: arr = [1024,512,256,128,64,32,16,8,4,2,1]
Output: [1,2,4,8,16,32,64,128,256,512,1024]
Explantion: All integers have 1 bit in the binary representation, you should just sort them in ascending order.
```

 

**Constraints:**

- `1 <= arr.length <= 500`
- `0 <= arr[i] <= 10 ** 4`

```python
def sortByBits(self, arr: List[int]) -> List[int]:
    return sorted(arr, key = lambda x: [bin(x).count('1'), x])
```

### 242. Valid Anagram

Given two strings `s` and `t`, return `true` *if* `t` *is an anagram of* `s`*, and* `false` *otherwise*.

An **Anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

 

**Example 1:**

```
Input: s = "anagram", t = "nagaram"
Output: true
```

**Example 2:**

```
Input: s = "rat", t = "car"
Output: false
```

 

**Constraints:**

- `1 <= s.length, t.length <= 5 * 104`
- `s` and `t` consist of lowercase English letters.

 

**Follow up:** What if the inputs contain Unicode characters? How would you adapt your solution to such a case?

```python
def isAnagram(self, s: str, t: str) -> bool:
    return sorted(s) == sorted(t)
```

### 1603. Design Parking System

Design a parking system for a parking lot. The parking lot has three kinds of parking spaces: big, medium, and small, with a fixed number of slots for each size.

Implement the `ParkingSystem` class:

- `ParkingSystem(int big, int medium, int small)` Initializes object of the `ParkingSystem` class. The number of slots for each parking space are given as part of the constructor.
- `bool addCar(int carType)` Checks whether there is a parking space of `carType` for the car that wants to get into the parking lot. `carType` can be of three kinds: big, medium, or small, which are represented by `1`, `2`, and `3` respectively. **A car can only park in a parking space of its** `carType`. If there is no space available, return `false`, else park the car in that size space and return `true`.

 

**Example 1:**

```
Input
["ParkingSystem", "addCar", "addCar", "addCar", "addCar"]
[[1, 1, 0], [1], [2], [3], [1]]
Output
[null, true, true, false, false]

Explanation
ParkingSystem parkingSystem = new ParkingSystem(1, 1, 0);
parkingSystem.addCar(1); // return true because there is 1 available slot for a big car
parkingSystem.addCar(2); // return true because there is 1 available slot for a medium car
parkingSystem.addCar(3); // return false because there is no available slot for a small car
parkingSystem.addCar(1); // return false because there is no available slot for a big car. It is already occupied.
```

 

**Constraints:**

- `0 <= big, medium, small <= 1000`
- `carType` is `1`, `2`, or `3`
- At most `1000` calls will be made to `addCar`

```python
class ParkingSystem:

    def __init__(self, big: int, medium: int, small: int):
        self.parking = {1: big, 2: medium, 3: small}

    def addCar(self, carType: int) -> bool:
        if self.parking[carType] > 0:
            self.parking[carType] -= 1
            return True
        return False
```

### 303. Range Sum Query - Immutable

Given an integer array `nums`, handle multiple queries of the following type:

1. Calculate the **sum** of the elements of `nums` between indices `left` and `right` **inclusive** where `left <= right`.

Implement the `NumArray` class:

- `NumArray(int[] nums)` Initializes the object with the integer array `nums`.
- `int sumRange(int left, int right)` Returns the **sum** of the elements of `nums` between indices `left` and `right` **inclusive** (i.e. `nums[left] + nums[left + 1] + ... + nums[right]`).

 

**Example 1:**

```
Input
["NumArray", "sumRange", "sumRange", "sumRange"]
[[[-2, 0, 3, -5, 2, -1]], [0, 2], [2, 5], [0, 5]]
Output
[null, 1, -1, -3]

Explanation
NumArray numArray = new NumArray([-2, 0, 3, -5, 2, -1]);
numArray.sumRange(0, 2); // return (-2) + 0 + 3 = 1
numArray.sumRange(2, 5); // return 3 + (-5) + 2 + (-1) = -1
numArray.sumRange(0, 5); // return (-2) + 0 + 3 + (-5) + 2 + (-1) = -3
```

 

**Constraints:**

- `1 <= nums.length <= 10 ** 4`
- `-10 ** 5 <= nums[i] <= 10 ** 5`
- `0 <= left <= right < nums.length`
- At most `10 ** 4` calls will be made to `sumRange`.

```python
class NumArray:

    def __init__(self, nums: List[int]):
        self.pre_sum = [0]
        for i in nums:
            self.pre_sum.append(self.pre_sum[-1] + i)

    def sumRange(self, left: int, right: int) -> int:
        return self.pre_sum[right + 1] - self.pre_sum[left]      
        
# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(left,right)
```

