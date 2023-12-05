### 2347. Best Poker Hand

You are given an integer array `ranks` and a character array `suits`. You have `5` cards where the `ith` card has a rank of `ranks[i]` and a suit of `suits[i]`.

The following are the types of **poker hands** you can make from best to worst:

1. `"Flush"`: Five cards of the same suit.
2. `"Three of a Kind"`: Three cards of the same rank.
3. `"Pair"`: Two cards of the same rank.
4. `"High Card"`: Any single card.

Return *a string representing the **best** type of **poker hand** you can make with the given cards.*

**Note** that the return values are **case-sensitive**.

 

**Example 1:**

```
Input: ranks = [13,2,3,1,9], suits = ["a","a","a","a","a"]
Output: "Flush"
Explanation: The hand with all the cards consists of 5 cards with the same suit, so we have a "Flush".
```

**Example 2:**

```
Input: ranks = [4,4,2,4,4], suits = ["d","a","a","b","c"]
Output: "Three of a Kind"
Explanation: The hand with the first, second, and fourth card consists of 3 cards with the same rank, so we have a "Three of a Kind".
Note that we could also make a "Pair" hand but "Three of a Kind" is a better hand.
Also note that other cards could be used to make the "Three of a Kind" hand.
```

**Example 3:**

```
Input: ranks = [10,10,2,12,9], suits = ["a","b","c","a","d"]
Output: "Pair"
Explanation: The hand with the first and second card consists of 2 cards with the same rank, so we have a "Pair".
Note that we cannot make a "Flush" or a "Three of a Kind".
```

 

**Constraints:**

- `ranks.length == suits.length == 5`
- `1 <= ranks[i] <= 13`
- `'a' <= suits[i] <= 'd'`
- No two cards have the same rank and suit.

```python
def bestHand(self, ranks: List[int], suits: List[str]) -> str:
    if len(set(suits)) == 1: return "Flush"
    cnt = collections.Counter(ranks)
    ret = max(cnt.values())
    if ret >= 3: return "Three of a Kind"
    if ret == 2: return "Pair"
    return "High Card"
```



### 2348. Number of Zero-Filled Subarrays

Given an integer array `nums`, return *the number of **subarrays** filled with* `0`.

A **subarray** is a contiguous non-empty sequence of elements within an array.

 

**Example 1:**

```
Input: nums = [1,3,0,0,2,0,0,4]
Output: 6
Explanation: 
There are 4 occurrences of [0] as a subarray.
There are 2 occurrences of [0,0] as a subarray.
There is no occurrence of a subarray with a size more than 2 filled with 0. Therefore, we return 6.
```

**Example 2:**

```
Input: nums = [0,0,0,2,0,0]
Output: 9
Explanation:
There are 5 occurrences of [0] as a subarray.
There are 3 occurrences of [0,0] as a subarray.
There is 1 occurrence of [0,0,0] as a subarray.
There is no occurrence of a subarray with a size more than 3 filled with 0. Therefore, we return 9.
```

**Example 3:**

```
Input: nums = [2,10,2019]
Output: 0
Explanation: There is no subarray filled with 0. Therefore, we return 0.
```

 

**Constraints:**

- `1 <= nums.length <= 10 ** 5`
- `-10 ** 9 <= nums[i] <= 10 ** 9`

```python
def zeroFilledSubarray(self, nums: List[int]) -> int:
    ret = cnt = 0
    nums.append(1)
    for i in nums:
        if i == 0: cnt += 1
        else:
            ret += cnt * (cnt + 1) // 2
            cnt = 0
    return ret
```



### 2349. Design a Number Container System

Design a number container system that can do the following:

- **Insert** or **Replace** a number at the given index in the system.
- **Return** the smallest index for the given number in the system.

Implement the `NumberContainers` class:

- `NumberContainers()` Initializes the number container system.
- `void change(int index, int number)` Fills the container at `index` with the `number`. If there is already a number at that `index`, replace it.
- `int find(int number)` Returns the smallest index for the given `number`, or `-1` if there is no index that is filled by `number` in the system.

 

**Example 1:**

```
Input
["NumberContainers", "find", "change", "change", "change", "change", "find", "change", "find"]
[[], [10], [2, 10], [1, 10], [3, 10], [5, 10], [10], [1, 20], [10]]
Output
[null, -1, null, null, null, null, 1, null, 2]

Explanation
NumberContainers nc = new NumberContainers();
nc.find(10); // There is no index that is filled with number 10. Therefore, we return -1.
nc.change(2, 10); // Your container at index 2 will be filled with number 10.
nc.change(1, 10); // Your container at index 1 will be filled with number 10.
nc.change(3, 10); // Your container at index 3 will be filled with number 10.
nc.change(5, 10); // Your container at index 5 will be filled with number 10.
nc.find(10); // Number 10 is at the indices 1, 2, 3, and 5. Since the smallest index that is filled with 10 is 1, we return 1.
nc.change(1, 20); // Your container at index 1 will be filled with number 20. Note that index 1 was filled with 10 and then replaced with 20. 
nc.find(10); // Number 10 is at the indices 2, 3, and 5. The smallest index that is filled with 10 is 2. Therefore, we return 2.
```

 

**Constraints:**

- `1 <= index, number <= 10 ** 9`
- At most `10 ** 5` calls will be made **in total** to `change` and `find`.

```python
import collections
import heapq

class NumberContainers:

    def __init__(self):
        self.val = collections.defaultdict(int)
        self.con = collections.defaultdict(list)

    def change(self, index: int, number: int) -> None:
        self.val[index] = number
        heapq.heappush(self.con[number], index)

    def find(self, number: int) -> int:
        idx = self.con[number]
        while idx and self.val[idx[0]] != number:
            heapq.heappop(idx)
        return idx[0] if idx else -1

# Your NumberContainers object will be instantiated and called as such:
# obj = NumberContainers()
# obj.change(index,number)
# param_2 = obj.find(number)
```



### 2350. Shortest Impossible Sequence of Rolls

You are given an integer array `rolls` of length `n` and an integer `k`. You roll a `k` sided dice numbered from `1` to `k`, `n` times, where the result of the `ith` roll is `rolls[i]`.

Return *the length of the **shortest** sequence of rolls that **cannot** be taken from* `rolls`.

A **sequence of rolls** of length `len` is the result of rolling a `k` sided dice `len` times.

**Note** that the sequence taken does not have to be consecutive as long as it is in order.

 

**Example 1:**

```
Input: rolls = [4,2,1,2,3,3,2,4,1], k = 4
Output: 3
Explanation: Every sequence of rolls of length 1, [1], [2], [3], [4], can be taken from rolls.
Every sequence of rolls of length 2, [1, 1], [1, 2], ..., [4, 4], can be taken from rolls.
The sequence [1, 4, 2] cannot be taken from rolls, so we return 3.
Note that there are other sequences that cannot be taken from rolls.
```

**Example 2:**

```
Input: rolls = [1,1,2,2], k = 2
Output: 2
Explanation: Every sequence of rolls of length 1, [1], [2], can be taken from rolls.
The sequence [2, 1] cannot be taken from rolls, so we return 2.
Note that there are other sequences that cannot be taken from rolls but [2, 1] is the shortest.
```

**Example 3:**

```
Input: rolls = [1,1,3,2,2,2,3,3], k = 4
Output: 1
Explanation: The sequence [4] cannot be taken from rolls, so we return 1.
Note that there are other sequences that cannot be taken from rolls but [4] is the shortest.
```

 

**Constraints:**

- `n == rolls.length`
- `1 <= n <= 10 ** 5`
- `1 <= rolls[i] <= k <= 10 ** 5`

```python
def shortestSequence(self, rolls: List[int], k: int) -> int:
    visited = set()
    ret = 0
    for i in rolls:
        visited.add(i)
        if len(visited) == k:
            ret += 1
            visited.clear()
    return ret + 1
```

