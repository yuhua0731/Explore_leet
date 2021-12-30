### 592. Fraction Addition and Subtraction

Given a string `expression` representing an expression of fraction addition and subtraction, return the calculation result in string format.

The final result should be an [irreducible fraction](https://en.wikipedia.org/wiki/Irreducible_fraction). If your final result is an integer, change it to the format of a fraction that has a denominator `1`. So in this case, `2` should be converted to `2/1`.

 

**Example 1:**

```
Input: expression = "-1/2+1/2"
Output: "0/1"
```

**Example 2:**

```
Input: expression = "-1/2+1/2+1/3"
Output: "1/3"
```

**Example 3:**

```
Input: expression = "1/3-1/2"
Output: "-1/6"
```

 

**Constraints:**

- The input string only contains `'0'` to `'9'`, `'/'`, `'+'` and `'-'`. So does the output.
- Each fraction (input and output) has the format `±numerator/denominator`. If the first input fraction or the output is positive, then `'+'` will be omitted.
- The input only contains valid **irreducible fractions**, where the **numerator** and **denominator** of each fraction will always be in the range `[1, 10]`. If the denominator is `1`, it means this fraction is actually an integer in a fraction format defined above.
- The number of given fractions will be in the range `[1, 10]`.
- The numerator and denominator of the **final result** are guaranteed to be valid and in the range of **32-bit** int.

#### My approach:

```python
def fractionAddition(self, expression: str) -> str:
    if expression[0] != '-': expression = '+' + expression
    m = re.findall('[+-][\d]+\/[\d]+', expression)
    sub = list()
    overall_lcm = 1
    def gcd(x, y):
        while y: x, y = y, x % y
        return x

    def lcm(x, y):
        return abs(x * y) // gcd(x, y)

    for i in m:
        fraction = list(map(int, i[1:].split('/')))
        sub.append([i[0], fraction[0], fraction[1]])
        overall_lcm = lcm(overall_lcm, fraction[1])
    print(sub)
    for i in range(len(sub)):
        sub[i][1] *= overall_lcm // sub[i][2]
    print(sub)
    overall_nomi = sum([-nomi if sign == '-' else nomi for sign, nomi, _ in sub])
    ans = '-' if overall_nomi < 0 else ''
    overall_nomi = abs(overall_nomi)
    g = gcd(overall_lcm, overall_nomi)
    return ans + f'{overall_nomi // g}/{overall_lcm // g}'
```

Too complicated and hard to read

#### Discussion

```python
m = map(int, re.findall('[+-]?[\d]+', expression))
    nomi, deno = 0, 1 # initialize result to 0/1
    for i in m:
        n = next(m)
        nomi = nomi * n + i * deno
        deno *= n
        g = math.gcd(nomi, deno)
        nomi //= g
        deno //= g
    return f'{nomi}/{deno}'
```

### 1920. Build Array from Permutation

Given a **zero-based permutation** `nums` (**0-indexed**), build an array `ans` of the **same length** where `ans[i] = nums[nums[i]]` for each `0 <= i < nums.length` and return it.

A **zero-based permutation** `nums` is an array of **distinct** integers from `0` to `nums.length - 1` (**inclusive**).

 

**Example 1:**

```
Input: nums = [0,2,1,5,3,4]
Output: [0,1,2,4,5,3]
Explanation: The array ans is built as follows: 
ans = [nums[nums[0]], nums[nums[1]], nums[nums[2]], nums[nums[3]], nums[nums[4]], nums[nums[5]]]
    = [nums[0], nums[2], nums[1], nums[5], nums[3], nums[4]]
    = [0,1,2,4,5,3]
```

**Example 2:**

```
Input: nums = [5,0,1,2,3,4]
Output: [4,5,0,1,2,3]
Explanation: The array ans is built as follows:
ans = [nums[nums[0]], nums[nums[1]], nums[nums[2]], nums[nums[3]], nums[nums[4]], nums[nums[5]]]
    = [nums[5], nums[0], nums[1], nums[2], nums[3], nums[4]]
    = [4,5,0,1,2,3]
```

 

**Constraints:**

- `1 <= nums.length <= 1000`
- `0 <= nums[i] < nums.length`
- The elements in `nums` are **distinct**.

 

**Follow-up:** Can you solve it without using an extra space (i.e., `O(1)` memory)?

#### basic approach:

```python
def buildArray(self, nums: List[int]) -> List[int]:
    return [nums[nums[i]] for i in range(len(nums))]
```

#### Follow-up requirement:

To prevent from using extra space, we have to modify the original list. the difficult point is, you have to use one value to record two values: original value and result value.

###### Note this line:

A **zero-based permutation** `nums` is an array of **distinct** integers from `0` to `nums.length - 1` (**inclusive**).

This means that all elements(original values & result values) in the input list are in range of [0, len(nums))

Thus, we can apply this expression to each element: `final_value = len(nums) * result_value + original_value`

- to retrieve result_value: `final_value // len(nums)`
- to retrieve original_value: `final_value % len(nums)`

```python
def buildArray(self, nums: List[int]) -> List[int]:
    # return [nums[nums[i]] for i in range(len(nums))]
    n = len(nums)
    for i in range(n): nums[i] = n * (nums[nums[i]] % n) + nums[i]
    for i in range(n): nums[i] //= n
    return nums
```

- Why do we choose `final_value = len(nums) * result_value + original_value` rather than `final_value = len(nums) * original_value + result_value`
  - note that when we calculate final value for i^th^ element, we have to retrieve `nums[nums[i]] % n`. We don’t need to care about if nums[nums[i]] has been calculated already or not, since original value is the ==remainder==, and will always keep unchanged.

### 13. Roman to Integer

Roman numerals are represented by seven different symbols: `I`, `V`, `X`, `L`, `C`, `D` and `M`.

```
Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```

For example, `2` is written as `II` in Roman numeral, just two one's added together. `12` is written as `XII`, which is simply `X + II`. The number `27` is written as `XXVII`, which is `XX + V + II`.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not `IIII`. Instead, the number four is written as `IV`. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as `IX`. There are six instances where subtraction is used:

- `I` can be placed before `V` (5) and `X` (10) to make 4 and 9. 
- `X` can be placed before `L` (50) and `C` (100) to make 40 and 90. 
- `C` can be placed before `D` (500) and `M` (1000) to make 400 and 900.

Given a roman numeral, convert it to an integer.

 

**Example 1:**

```
Input: s = "III"
Output: 3
Explanation: III = 3.
```

**Example 2:**

```
Input: s = "LVIII"
Output: 58
Explanation: L = 50, V= 5, III = 3.
```

**Example 3:**

```
Input: s = "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
```

 

**Constraints:**

- `1 <= s.length <= 15`
- `s` contains only the characters `('I', 'V', 'X', 'L', 'C', 'D', 'M')`.
- It is **guaranteed** that `s` is a valid roman numeral in the range `[1, 3999]`.

