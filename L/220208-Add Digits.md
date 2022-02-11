Given an integer `num`, repeatedly add all its digits until the result has only one digit, and return it.

 

**Example 1:**

```
Input: num = 38
Output: 2
Explanation: The process is
38 --> 3 + 8 --> 11
11 --> 1 + 1 --> 2 
Since 2 has only one digit, return it.
```

**Example 2:**

```
Input: num = 0
Output: 0
```

 

**Constraints:**

- `0 <= num <= 231 - 1`

 

**Follow up:** Could you do it without any loop/recursion in `O(1)` runtime?



Find the rule of result

```python
# Input from 0 to 30
for i in range(30):
    while i > 9:
        i = eval('+'.join(list(str(i))))
    print(i, end='')
# Output:
# 012345678912345678912345678912

```

> except for 0, `result = [num % 9, 9][num % 9 == 0]`

```python
def addDigits(self, num: int) -> int:
    return 0 if not num else [num % 9, 9][num % 9 == 0]
```

