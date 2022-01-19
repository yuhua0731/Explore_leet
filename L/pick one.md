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

```python
def romanToInt(self, s: str) -> int:
    """
    special sequences: IV(4) IX(9) XL(40) XC(90) CD(400) CM(900)
    general chars: I(1) V(5) X(10) L(50) C(100) D(500) M(1000)
    """
    # dp
    pre_2, pre_1 = 0, 0
    pre_c = ' '
    special = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000,
               'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90, 'CD': 400, 'CM': 900}
    for c in s:
        if pre_c + c in special:
            pre_2, pre_1 = pre_1, pre_2 + special[pre_c + c]
        else:
            pre_2, pre_1 = pre_1, pre_1 + special[c]
        pre_c = c
    return pre_1
```

### 1834. Single-Threaded CPU

You are given `n` tasks labeled from `0` to `n - 1` represented by a 2D integer array `tasks`, where `tasks[i] = [enqueueTimei, processingTimei]` means that the `ith` task will be available to process at `enqueueTimei` and will take `processingTimei` to finish processing.

You have a single-threaded CPU that can process **at most one** task at a time and will act in the following way:

- If the CPU is idle and there are no available tasks to process, the CPU remains idle.
- If the CPU is idle and there are available tasks, the CPU will choose the one with the **shortest processing time**. If multiple tasks have the same shortest processing time, it will choose the task with the smallest index.
- Once a task is started, the CPU will **process the entire task** without stopping.
- The CPU can finish a task then start a new one instantly.

Return *the order in which the CPU will process the tasks.*

 

**Example 1:**

```
Input: tasks = [[1,2],[2,4],[3,2],[4,1]]
Output: [0,2,3,1]
Explanation: The events go as follows: 
- At time = 1, task 0 is available to process. Available tasks = {0}.
- Also at time = 1, the idle CPU starts processing task 0. Available tasks = {}.
- At time = 2, task 1 is available to process. Available tasks = {1}.
- At time = 3, task 2 is available to process. Available tasks = {1, 2}.
- Also at time = 3, the CPU finishes task 0 and starts processing task 2 as it is the shortest. Available tasks = {1}.
- At time = 4, task 3 is available to process. Available tasks = {1, 3}.
- At time = 5, the CPU finishes task 2 and starts processing task 3 as it is the shortest. Available tasks = {1}.
- At time = 6, the CPU finishes task 3 and starts processing task 1. Available tasks = {}.
- At time = 10, the CPU finishes task 1 and becomes idle.
```

**Example 2:**

```
Input: tasks = [[7,10],[7,12],[7,5],[7,4],[7,2]]
Output: [4,3,2,0,1]
Explanation: The events go as follows:
- At time = 7, all the tasks become available. Available tasks = {0,1,2,3,4}.
- Also at time = 7, the idle CPU starts processing task 4. Available tasks = {0,1,2,3}.
- At time = 9, the CPU finishes task 4 and starts processing task 3. Available tasks = {0,1,2}.
- At time = 13, the CPU finishes task 3 and starts processing task 2. Available tasks = {0,1}.
- At time = 18, the CPU finishes task 2 and starts processing task 0. Available tasks = {1}.
- At time = 28, the CPU finishes task 0 and starts processing task 1. Available tasks = {}.
- At time = 40, the CPU finishes task 1 and becomes idle.
```

 

**Constraints:**

- `tasks.length == n`
- `1 <= n <= 105`
- `1 <= enqueueTimei, processingTimei <= 109`

My approach:

1. Sort the input list, key = available time, process time, original index
2. create an heapq: store all available tasks at time i, element = (process time, index)
3. Pop a task from heapq, and update curr time. return answer list until all tasks has been added into it.

```python
def getOrder(self, tasks: List[List[int]]) -> List[int]:
    # heapq element: (process time, index)
    pq = []
    tasks = sorted([[task[0], task[1], index] for index, task in enumerate(tasks)])
    ans = list()
    curr_time = 0
    while tasks or pq:
        # if heapq is empty, we must add the first tasks into it, and update curr_time if needed
        if not pq: curr_time = max(tasks[0][0], curr_time)
        while tasks and tasks[0][0] <= curr_time:
            temp = tasks.pop(0)
            heapq.heappush(pq, (temp[1], temp[2]))
        # pq is not empty now
        # pop out the first task
        process, index = heapq.heappop(pq)
        curr_time += process - 1
        ans.append(index)
    return ans
```

### 403. Frog Jump

A frog is crossing a river. The river is divided into some number of units, and at each unit, there may or may not exist a stone. The frog can jump on a stone, but it must not jump into the water.

Given a list of `stones`' positions (in units) in sorted **ascending order**, determine if the frog can cross the river by landing on the last stone. Initially, the frog is on the first stone and assumes the first jump must be `1` unit.

If the frog's last jump was `k` units, its next jump must be either `k - 1`, `k`, or `k + 1` units. The frog can only jump in the ==forward== direction.

 

**Example 1:**

```
Input: stones = [0,1,3,5,6,8,12,17]
Output: true
Explanation: The frog can jump to the last stone by jumping 1 unit to the 2nd stone, then 2 units to the 3rd stone, then 2 units to the 4th stone, then 3 units to the 6th stone, 4 units to the 7th stone, and 5 units to the 8th stone.
```

**Example 2:**

```
Input: stones = [0,1,2,3,4,8,9,11]
Output: false
Explanation: There is no way to jump to the last stone as the gap between the 5th and 6th stone is too large.
```

 

**Constraints:**

- `2 <= stones.length <= 2000`
- `0 <= stones[i] <= 2 ** 31 - 1`
- `stones[0] == 0`
- `stones` is sorted in a strictly increasing order.

#### My approach:

simulate how frog jumps, using recursion

1. implement a function: take current position and last jump step as input
2. return true if frog can reach last stone from current state
3. to improve performance, we pruncate recursion calls with a set called `visited`. If same input combination is detected, return False directly.

```python
def canCross(self, stones: List[int]) -> bool:
    visited = set()
    def jump(curr, k):
        if (curr, k) in visited: return False
        visited.add((curr, k))
        if curr not in stones: return False
        if curr == stones[-1]: return True
        # curr position in stones, jump to next place
        step = [k - 1, k, k + 1]
        return any(jump(curr + i, i) for i in step)
    return jump(1, 1) if stones[1] == 1 else False
```

#### Discussion approach:

Recursion => DP

1. create a 2-D array DP. `dp[i][j]` represents for the possibility of make a j step jump at stones[i] position.
2. dp array’s size: since we can increase our step by 1 at most every time, for example, frog can jump at most 1 at stone[0], and at most 2 at stones[1], .., when frog reach stones[i], he could jump i + 1 at most, hence, dp can be initialized to `[[0] * (n + 1) for _ in range(n)]`.
3. iterate on stones, for stones[i], check all stones[j] (j < i), if frog can jump from stones[j] to stones[i].
4. return `any(dp[-1])`.

```python
def canCross(self, stones: List[int]) -> bool:
    # dp
    n = len(stones)
    dp = [[0] * (n + 1) for _ in range(n)]
    dp[0][1] = 1 # frog can only jump 1 step at stones[0] = 0
    for i in range(1, n):
        for j in range(i - 1, -1, -1): # from right to left to allow us prune iteration
            step = stones[i] - stones[j]
            if step > j + 1: break # frog can jump at most j + 1 step at stones[j]
            if dp[j][step]: dp[i][step - 1] = dp[i][step] = dp[i][step + 1] = 1
    return any(dp[-1])
```

4 times faster than recursion, while still perform terrible in time complexity

### 1801. Number of Orders in the Backlog

You are given a 2D integer array `orders`, where each `orders[i] = [pricei, amounti, orderTypei]` denotes that `amounti` orders have been placed of type `orderTypei` at the price `pricei`. The `orderTypei` is:

- `0` if it is a batch of `buy` orders, or
- `1` if it is a batch of `sell` orders.

Note that `orders[i]` represents a batch of `amounti` independent orders with the same price and order type. All orders represented by `orders[i]` will be placed before all orders represented by `orders[i+1]` for all valid `i`.

There is a **backlog** that consists of orders that have not been executed. The backlog is initially empty. When an order is placed, the following happens:

- If the order is a `buy` order, you look at the `sell` order with the **smallest** price in the backlog. If that `sell` order's price is **smaller than or equal to** the current `buy` order's price, they will match and be executed, and that `sell` order will be removed from the backlog. Else, the `buy` order is added to the backlog.
- Vice versa, if the order is a `sell` order, you look at the `buy` order with the **largest** price in the backlog. If that `buy` order's price is **larger than or equal to** the current `sell` order's price, they will match and be executed, and that `buy` order will be removed from the backlog. Else, the `sell` order is added to the backlog.

Return *the total **amount** of orders in the backlog after placing all the orders from the input*. Since this number can be large, return it **modulo** `10 ** 9 + 7`.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/03/11/ex1.png)

```
Input: orders = [[10,5,0],[15,2,1],[25,1,1],[30,4,0]]
Output: 6
Explanation: Here is what happens with the orders:
- 5 orders of type buy with price 10 are placed. There are no sell orders, so the 5 orders are added to the backlog.
- 2 orders of type sell with price 15 are placed. There are no buy orders with prices larger than or equal to 15, so the 2 orders are added to the backlog.
- 1 order of type sell with price 25 is placed. There are no buy orders with prices larger than or equal to 25 in the backlog, so this order is added to the backlog.
- 4 orders of type buy with price 30 are placed. The first 2 orders are matched with the 2 sell orders of the least price, which is 15 and these 2 sell orders are removed from the backlog. The 3rd order is matched with the sell order of the least price, which is 25 and this sell order is removed from the backlog. Then, there are no more sell orders in the backlog, so the 4th order is added to the backlog.
Finally, the backlog has 5 buy orders with price 10, and 1 buy order with price 30. So the total number of orders in the backlog is 6.
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2021/03/11/ex2.png)

```
Input: orders = [[7,1000000000,1],[15,3,0],[5,999999995,0],[5,1,1]]
Output: 999999984
Explanation: Here is what happens with the orders:
- 109 orders of type sell with price 7 are placed. There are no buy orders, so the 109 orders are added to the backlog.
- 3 orders of type buy with price 15 are placed. They are matched with the 3 sell orders with the least price which is 7, and those 3 sell orders are removed from the backlog.
- 999999995 orders of type buy with price 5 are placed. The least price of a sell order is 7, so the 999999995 orders are added to the backlog.
- 1 order of type sell with price 5 is placed. It is matched with the buy order of the highest price, which is 5, and that buy order is removed from the backlog.
Finally, the backlog has (1000000000-3) sell orders with price 7, and (999999995-1) buy orders with price 5. So the total number of orders = 1999999991, which is equal to 999999984 % (109 + 7).
```

 

**Constraints:**

- `1 <= orders.length <= 105`
- `orders[i].length == 3`
- `1 <= pricei, amounti <= 109`
- `orderTypei` is either `0` or `1`.

```python
def getNumberOfBacklogOrders(self, orders: List[List[int]]) -> int:
    # you must obey the original order sequence
    # heapq
    # sell's element = (price, amount)
    # buy's element = (-price, amount)
    buy, sell = [], []
    for price, amount, sell_buy in orders:
        if sell_buy:
            heapq.heappush(sell, [price, amount])
        else:
            heapq.heappush(buy, [-price, amount])

        while buy and sell and -buy[0][0] >= sell[0][0]:
            k = min(buy[0][1], sell[0][1])
            buy[0][1] -= k
            sell[0][1] -= k
            if buy[0][1] == 0: heapq.heappop(buy)
            if sell[0][1] == 0: heapq.heappop(sell)
    return sum(amount for _, amount in buy + sell) % (10 ** 9 + 7)
```

### 1606. Find Servers That Handled Most Number of Requests

You have `k` servers numbered from `0` to `k-1` that are being used to handle multiple requests simultaneously. Each server has infinite computational capacity but **cannot handle more than one request at a time**. The requests are assigned to servers according to a specific algorithm:

- The `ith` (0-indexed) request arrives.
- If all servers are busy, the request is dropped (not handled at all).
- If the `(i % k)th` server is available, assign the request to that server.
- Otherwise, assign the request to the next available server (wrapping around the list of servers and starting from 0 if necessary). For example, if the `ith` server is busy, try to assign the request to the `(i+1)th` server, then the `(i+2)th` server, and so on.

You are given a **strictly increasing** array `arrival` of positive integers, where `arrival[i]` represents the arrival time of the `ith` request, and another array `load`, where `load[i]` represents the load of the `ith` request (the time it takes to complete). Your goal is to find the **busiest server(s)**. A server is considered **busiest** if it handled the most number of requests successfully among all the servers.

Return *a list containing the IDs (0-indexed) of the **busiest server(s)***. You may return the IDs in any order.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/09/08/load-1.png)

```
Input: k = 3, arrival = [1,2,3,4,5], load = [5,2,3,3,3] 
Output: [1] 
Explanation: 
All of the servers start out available.
The first 3 requests are handled by the first 3 servers in order.
Request 3 comes in. Server 0 is busy, so it's assigned to the next available server, which is 1.
Request 4 comes in. It cannot be handled since all servers are busy, so it is dropped.
Servers 0 and 2 handled one request each, while server 1 handled two requests. Hence server 1 is the busiest server.
```

**Example 2:**

```
Input: k = 3, arrival = [1,2,3,4], load = [1,2,1,2]
Output: [0]
Explanation: 
The first 3 requests are handled by first 3 servers.
Request 3 comes in. It is handled by server 0 since the server is available.
Server 0 handled two requests, while servers 1 and 2 handled one request each. Hence server 0 is the busiest server.
```

**Example 3:**

```
Input: k = 3, arrival = [1,2,3], load = [10,12,11]
Output: [0,1,2]
Explanation: Each server handles a single request, so they are all considered the busiest.
```

 

**Constraints:**

- `1 <= k <= 10 ** 5`
- `1 <= arrival.length, load.length <= 10 ** 5`
- `arrival.length == load.length`
- `1 <= arrival[i], load[i] <= 10 ** 9`
- `arrival` is **strictly increasing**.

#### My approach:

using two lists, and got TLE with huge input size

```python
def busiestServers(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
    ans = [0] * k # ans[i] represents for the amount of requests ith server handled
    free_time = [-1] * k # free_time represents for the time that ith server will be free to take request

    for idx, request in enumerate(zip(arrival, load)):
        start, end = request[0], sum(request)
        for i in range(idx, idx + k):
            i_server = i % k
            if start >= free_time[i_server]:
                free_time[i_server] = end
                ans[i_server] += 1
                break

    ret = list()
    most = 0
    for idx, cnt in enumerate(ans):
        if cnt > most: most, ret = cnt, [idx]
        elif cnt == most: ret.append(idx)
    return ret
```

#### Discussion:

Instead of using list, think about heap, which keep its elements sorted all the time. and do not require you to loop manually.

##### ==three-heap== version

easy to understand, but have messy code

```python
def busiestServers(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
    ans = [0] * k # ans[i] represents for the amount of requests ith server handled
    busy = [] # heap: servers that currently occupied by a request, element = (free_time, idx)
    free_behind = [] # heap: servers that currently free to handle request, idx equal or greater than i, element = (idx)
    free_ahead = [i for i in range(k)] # heap: servers that currently free to handle request, idx less than i
    for idx, (start, last) in enumerate(zip(arrival, load)):
        idx %= k
        if idx == 0:
            free_behind = free_ahead
            free_ahead = []
        while busy and busy[0][0] <= start:
            _, i = heapq.heappop(busy)
            if i >= idx: heapq.heappush(free_behind, i)
            else: heapq.heappush(free_ahead, i)
        while free_behind and free_behind[0] < idx:
            heapq.heappush(free_ahead, heapq.heappop(free_behind))
        use_heap = free_behind if free_behind else free_ahead
        if use_heap:
            assign =  heapq.heappop(use_heap)
            heapq.heappush(busy, (start + last, assign))
            ans[assign] += 1
    most = max(ans)
    return [i for i, cnt in enumerate(ans) if cnt == most]
```

##### ==two-heap== version

really genius while a little bit hard to understand

```python
def busiestServers(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
    # two-heap
    ans = [0] * k # ans[i] represents for the amount of requests ith server handled
    busy = [] # heap: servers that currently occupied by a request, element = (free_time, idx)
    free = [i for i in range(k)] # heap: servers that currently free to handle request, element = (idx)
    for idx, (start, last) in enumerate(zip(arrival, load)):
        while busy and busy[0][0] <= start:
            _, i = heapq.heappop(busy)
            # instead of push i into free, we record idx + (i - idx) % k here as the server index
            # believe it or not, this magic expression ensure we sort server in idx % k -> k - 1 -> 0 -> idx % k - 1 order
            # remainder = - 1 - idx + k * divisor = - 1 + (k * divisor - idx) = - 1 + k - idx % k
            heapq.heappush(free, idx + (i - idx) % k)
        if free:
            assign = heapq.heappop(free) % k
            heapq.heappush(busy, (start + last, assign))
            ans[assign] += 1
    most = max(ans)
    return [i for i, cnt in enumerate(ans) if cnt == most]
```

To merge two heaps into one, we cannot simply push index of free server into it. Our goal is: sort server in the following order:

`idx % k —> k - 1 ——(wrap list)——> 0 -> idx % k - 1`

here we use an expression: `idx + (i - idx) % k`, the following table evaluate how it works:

| i           | expression: idx + (i - idx) % k | record in heap free   | order: range(idx, idx + k) |
| ----------- | ------------------------------- | --------------------- | -------------------------- |
| idx % k     | idx + (idx % k - idx) % k       | idx                   | idx + k - k                |
| idx % k + 1 | idx + (idx % k + 1 - idx) % k   | idx + 1               | idx + k - (k - 1)          |
| k - 1       | idx + (k - 1 - idx) % k         | idx + k - idx % k - 1 | idx + k - (idx % k + 1)    |
| 0           | idx + (- idx) % k               | idx + k - idx % k     | idx + k - (idx % k)        |
| idx % k - 1 | idx + (idx % k - 1 - idx) % k   | idx + (- 1) % k       | idx + k - 1                |

Confusing part: `i = k - 1`
$$
remainder = (k-1-idx)\%k = (-1-idx)\%k = (-1)\%k+(-idx)\%k\\
=(k-1)+(k-idx\%k)=k-idx\%k-1
$$

### 761. Special Binary String

**Special binary strings** are binary strings with the following two properties:

- The number of `0`'s is equal to the number of `1`'s.
- Every prefix of the binary string has at least as many `1`'s as `0`'s.

You are given a **special binary** string `s`.

A move consists of choosing two consecutive, non-empty, special substrings of `s`, and swapping them. Two strings are consecutive if the last character of the first string is exactly one index before the first character of the second string.

Return *the lexicographically largest resulting string possible after applying the mentioned operations on the string*.

 

**Example 1:**

```
Input: s = "11011000"
Output: "11100100"
Explanation: The strings "10" [occuring at s[1]] and "1100" [at s[3]] are swapped.
This is the lexicographically largest string possible after some number of swaps.
```

**Example 2:**

```
Input: s = "10"
Output: "10"
```

 

**Constraints:**

- `1 <= s.length <= 50`
- `s[i]` is either `'0'` or `'1'`.
- `s` is a special binary string.

#### First Approach:

> 1. 首先，定义==特殊字符串==：拥有同样数量的0和1、从前往后依次延长前缀子字符串，不能出现0比1多的情况。
> 2. 在输入的字符串中，找到两个连续的、非空的特殊字符串，互换位置，组成新的字符串。
> 3. 返回能够得到的最大的新字符串。
>
> 初始思路：
>
> 首先，这个特殊字符串，一定是以1打头的，以0结尾的。
>
> 找出所有可能的特殊字符串.
>
> 再找出其中连续的，能够成组的.
>
> 依次互换位置，返回能够得到的最大的新字符串。

```python
def makeLargestSpecial(self, s: str) -> str:
    special = list() # element: (start, end)
    curr = list() # element: [start_idx, cnt_0, cnt_1]
    for i in range(len(s)):
        for j in range(len(curr)):
            if s[i] == '0':
                curr[j][1] += 1
            else:
                curr[j][2] += 1
        if s[i] == '1': curr.append([i, 0, 1])
        curr = [[idx, x, y] for idx, x, y in curr if x <= y]
        special += [[idx, idx + x + y] for idx, x, y in curr if x == y]

    # find consecutive pairs
    special.sort()
    pair = []
    for i in range(len(special)):
        for j in range(i + 1, len(special)):
            if special[i][1] == special[j][0]: pair.append(special[i] + special[j])
            if special[i][1] < special[j][0]: break

    res = [s]
    for s1, e1, s2, e2 in pair: heapq.heappush(res, s[:s1] + s[s2:e2] + s[s1:e1] + s[e2:])
    return heapq.nlargest(1, res)[0]
```

misunderstood this question..

it doesn’t mention about how many moves we are allowed to do.

Actually, we can do ==as many moves as== we want to find the lexicographically largest string.

> new idea:
>
> 1. Split input string into ==as many== special strings as possible. Since input string is a special string. It is guarantted that we can split it into several substrings, and they are all special strings.(note: we must split the whole original string, and make sure all substrings are special strings. For instance, the string to be splited is “110100”, we can only get [“110100”] as result, rather than [“1”, “10”, “10”, “0”] since “1” & “0” are not special strings)
> 2. Sort all substrings in default order.
> 3. Join them in reverse order, then return it.

```python
def makeLargestSpecial(self, s: str) -> str:
    start = cnt = 0
    special = list()
    for i, c in enumerate(s):
        if c == '1': cnt += 1
        if c == '0': cnt -= 1
        if cnt == 0: 
            special.append(s[start:i + 1])
            start = i + 1
            cnt = 0
    return ''.join(sorted(special)[::-1])
```

This solution is wrong in some cases

There is one example testcase:

Input: “11011000”

Output: “11011000”

Expect: “11100100”

#### Discussion:

Think about this situation: we have a special string “11011000”, due to our split logic, we cannot further split it any more. However, there are two consecutive special strings “10” & “1100” in it, which can be swapped and should be swapped to get a larger new string.

After viewing Lee’s solution, I learned a solution for this case. Actually, for each special substring `subs`, we should proceed a recursive call for `subs[1:-1]`, making themselves become the lexicographically largest string.

```python
@cahce
def makeLargestSpecial(self, s: str) -> str:
    start = 0
    cnt = 0
    special = list()
    for i, c in enumerate(s):
        if c == '1': cnt += 1
        if c == '0': cnt -= 1
        if cnt == 0: 
            special.append('1' + self.makeLargestSpecial(s[start + 1:i]) + '0')
            start = i + 1
            cnt = 0
    return ''.join(sorted(special)[::-1])
```

Provement:

for a special string, it must obey the form “1M0” due to  the defination of special string.

It is also guarantted that the middle part M is also a speical string.

- since “1M0” satisfies the condition that cnt_0 == cnt_1, then M satisfies it obviously
- since we will stop extending substring once we find that cnt_0 == cnt_1, and we didn’t stop until we reached the last 0 in this substring. We can tell that for each prefix of “1M0”, cnt_1 > cnt_0.
- after we discarded the first ‘1’, cnt_1 >= cnt_0 is still guarantted for each prefix of M.

Therefore, M is also a special string and we can call `makeLargestSpecial(M)` to get the lexicographically largest string of M.

### 1745. Palindrome Partitioning IV

Given a string `s`, return `true` *if it is possible to split the string* `s` *into three **non-empty** palindromic substrings. Otherwise, return* `false`.

A string is said to be palindrome if it the same string when reversed.

 

**Example 1:**

```
Input: s = "abcbdd"
Output: true
Explanation: "abcbdd" = "a" + "bcb" + "dd", and all three substrings are palindromes.
```

**Example 2:**

```
Input: s = "bcbddxy"
Output: false
Explanation: s cannot be split into 3 palindromes.
```

 

**Constraints:**

- `3 <= s.length <= 2000`
- `s` consists only of lowercase English letters.

#### My approach

1. dp map is our fundation.
2. create a recursive function that help you split the original string into any amount of palindromes.
3. have fun!

```python
def checkPartitioning(self, s: str) -> bool:
    # dp[i][j] = True if s[i:j] is a palindrome
    n = len(s)
    dp = [[False] * (n + 1) for _ in range(n + 1)]
    for diff in range(n + 1):
        for i in range(n + 1 - diff):
            j = i + diff
            if diff <= 1: dp[i][j] = True
            elif s[i] == s[j - 1]: dp[i][j] = dp[i + 1][j - 1]

    def part(start: int, cnt: int):
        if start == n: return False
        if cnt == 1: return dp[start][n]
        return any(part(i + 1, cnt - 1) for i in range(start, n) if dp[start][i + 1])
```

### 1476. Subrectangle Queries

Implement the class `SubrectangleQueries` which receives a `rows x cols` rectangle as a matrix of integers in the constructor and supports two methods:

1.` updateSubrectangle(int row1, int col1, int row2, int col2, int newValue)`

- Updates all values with `newValue` in the subrectangle whose upper left coordinate is `(row1,col1)` and bottom right coordinate is `(row2,col2)`.

2.` getValue(int row, int col)`

- Returns the current value of the coordinate `(row,col)` from the rectangle.

 

**Example 1:**

```
Input
["SubrectangleQueries","getValue","updateSubrectangle","getValue","getValue","updateSubrectangle","getValue","getValue"]
[[[[1,2,1],[4,3,4],[3,2,1],[1,1,1]]],[0,2],[0,0,3,2,5],[0,2],[3,1],[3,0,3,2,10],[3,1],[0,2]]
Output
[null,1,null,5,5,null,10,5]
Explanation
SubrectangleQueries subrectangleQueries = new SubrectangleQueries([[1,2,1],[4,3,4],[3,2,1],[1,1,1]]);  
// The initial rectangle (4x3) looks like:
// 1 2 1
// 4 3 4
// 3 2 1
// 1 1 1
subrectangleQueries.getValue(0, 2); // return 1
subrectangleQueries.updateSubrectangle(0, 0, 3, 2, 5);
// After this update the rectangle looks like:
// 5 5 5
// 5 5 5
// 5 5 5
// 5 5 5 
subrectangleQueries.getValue(0, 2); // return 5
subrectangleQueries.getValue(3, 1); // return 5
subrectangleQueries.updateSubrectangle(3, 0, 3, 2, 10);
// After this update the rectangle looks like:
// 5   5   5
// 5   5   5
// 5   5   5
// 10  10  10 
subrectangleQueries.getValue(3, 1); // return 10
subrectangleQueries.getValue(0, 2); // return 5
```

**Example 2:**

```
Input
["SubrectangleQueries","getValue","updateSubrectangle","getValue","getValue","updateSubrectangle","getValue"]
[[[[1,1,1],[2,2,2],[3,3,3]]],[0,0],[0,0,2,2,100],[0,0],[2,2],[1,1,2,2,20],[2,2]]
Output
[null,1,null,100,100,null,20]
Explanation
SubrectangleQueries subrectangleQueries = new SubrectangleQueries([[1,1,1],[2,2,2],[3,3,3]]);
subrectangleQueries.getValue(0, 0); // return 1
subrectangleQueries.updateSubrectangle(0, 0, 2, 2, 100);
subrectangleQueries.getValue(0, 0); // return 100
subrectangleQueries.getValue(2, 2); // return 100
subrectangleQueries.updateSubrectangle(1, 1, 2, 2, 20);
subrectangleQueries.getValue(2, 2); // return 20
```

 

**Constraints:**

- There will be at most `500` operations considering both methods: `updateSubrectangle` and `getValue`.
- `1 <= rows, cols <= 100`
- `rows == rectangle.length`
- `cols == rectangle[i].length`
- `0 <= row1 <= row2 < rows`
- `0 <= col1 <= col2 < cols`
- `1 <= newValue, rectangle[i][j] <= 10^9`
- `0 <= row < rows`
- `0 <= col < cols`

```python
class SubrectangleQueries:

    def __init__(self, rectangle: List[List[int]]):
        self.rect = copy.deepcopy(rectangle)

    def updateSubrectangle(self, row1: int, col1: int, row2: int, col2: int, newValue: int) -> None:
        for i in range(row1, row2 + 1):
            for j in range(col1, col2 + 1):
                self.rect[i][j] = newValue

    def getValue(self, row: int, col: int) -> int:
        return self.rect[row][col]


if __name__ == '__main__':
    # Your SubrectangleQueries object will be instantiated and called as such:
    rectangle = [[1, 2, 1], [4, 3, 4], [3, 2, 1], [1, 1, 1]]
    obj = SubrectangleQueries(rectangle)
    obj.updateSubrectangle(0, 0, 3, 2, 5)
    print(obj.getValue(0, 2))
```

### 313. Super Ugly Number

A **super ugly number** is a positive integer whose prime factors are in the array `primes`.

Given an integer `n` and an array of integers `primes`, return *the* `nth` ***super ugly number***.

The `nth` **super ugly number** is **guaranteed** to fit in a **32-bit** signed integer.

 

**Example 1:**

```
Input: n = 12, primes = [2,7,13,19]
Output: 32
Explanation: [1,2,4,7,8,13,14,16,19,26,28,32] is the sequence of the first 12 super ugly numbers given primes = [2,7,13,19].
```

**Example 2:**

```
Input: n = 1, primes = [2,3,5]
Output: 1
Explanation: 1 has no prime factors, therefore all of its prime factors are in the array primes = [2,3,5].
```

 

**Constraints:**

- `1 <= n <= 10 ** 6`
- `1 <= primes.length <= 100`
- `2 <= primes[i] <= 1000`
- `primes[i]` is **guaranteed** to be a prime number.
- All the values of `primes` are **unique** and sorted in **ascending order**.

#### First Approach

using heapq, pop ugly numbers one by one.

this solution result in TLE when n is really huge.

```python
def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
    visited = set()
    primes_pq = primes.copy()
    temp = 1
    for _ in range(1, n):
        while primes_pq[0] in visited:
            heapq.heappop(primes_pq)
        temp = heapq.heappop(primes_pq)
        visited.add(temp)
        [heapq.heappush(primes_pq, temp * i) for i in primes]
    return temp
```

#### Second Approach

```python
def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:	
    res, heap = [1], [(primes[i], primes[i], 0) for i in range(len(primes))]
    while len(res) < n:
        val, prm, idx = heappop(heap)
        if val <= res[-1]:
            while val <= res[-1]: idx += 1; val = prm * res[idx]
        else:
            res += val,
            val, idx = prm * res[idx + 1], idx + 1
        heappush(heap, (val, prm, idx))
    return res[-1]
```

### 1958. Check if Move is Legal

You are given a **0-indexed** `8 x 8` grid `board`, where `board[r][c]` represents the cell `(r, c)` on a game board. On the board, free cells are represented by `'.'`, white cells are represented by `'W'`, and black cells are represented by `'B'`.

Each move in this game consists of choosing a free cell and changing it to the color you are playing as (either white or black). However, a move is only **legal** if, after changing it, the cell becomes the **==endpoint== of a good line** (horizontal, vertical, or diagonal).

A **good line** is a line of **three or more cells (including the endpoints)** where the endpoints of the line are **one color**, and the remaining cells in the middle are the **opposite color** (no cells in the line are free). You can find examples for good lines in the figure below:

![img](https://assets.leetcode.com/uploads/2021/07/22/goodlines5.png)

Given two integers `rMove` and `cMove` and a character `color` representing the color you are playing as (white or black), return `true` *if changing cell* `(rMove, cMove)` *to color* `color` *is a **legal** move, or* `false` *if it is not legal*.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/07/10/grid11.png)

```
Input: board = [[".",".",".","B",".",".",".","."],[".",".",".","W",".",".",".","."],[".",".",".","W",".",".",".","."],[".",".",".","W",".",".",".","."],["W","B","B",".","W","W","W","B"],[".",".",".","B",".",".",".","."],[".",".",".","B",".",".",".","."],[".",".",".","W",".",".",".","."]], rMove = 4, cMove = 3, color = "B"
Output: true
Explanation: '.', 'W', and 'B' are represented by the colors blue, white, and black respectively, and cell (rMove, cMove) is marked with an 'X'.
The two good lines with the chosen cell as an endpoint are annotated above with the red rectangles.
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2021/07/10/grid2.png)

```
Input: board = [[".",".",".",".",".",".",".","."],[".","B",".",".","W",".",".","."],[".",".","W",".",".",".",".","."],[".",".",".","W","B",".",".","."],[".",".",".",".",".",".",".","."],[".",".",".",".","B","W",".","."],[".",".",".",".",".",".","W","."],[".",".",".",".",".",".",".","B"]], rMove = 4, cMove = 4, color = "W"
Output: false
Explanation: While there are good lines with the chosen cell as a middle cell, there are no good lines with the chosen cell as an endpoint.
```

 

**Constraints:**

- `board.length == board[r].length == 8`
- `0 <= rMove, cMove < 8`
- `board[rMove][cMove] == '.'`
- `color` is either `'B'` or `'W'`.

#### My approach

1. Implement a function called extend, taking original position, extend direction, and endpoint color as argument
2. since we need to consider 8 directions totally, we just check if any direction in these 8-dir can extend to a good line.

```python
def checkMove(self, board: List[List[str]], rMove: int, cMove: int, color: str) -> bool:
    dirs = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
    # horizontal | vertical | diagonal

    def extend(x, y, dx, dy, end_color):
        i = 1
        while True:
            nx, ny = x + i * dx, y + i * dy
            if 8 > nx >= 0 <= ny < 8:
                if board[nx][ny] == '.': return False
                if board[nx][ny] == end_color:
                    return False if i == 1 else True
                i += 1
            else: return False

    return any(extend(rMove, cMove, i, j, color) for i, j in dirs)
```

### 1402. Reducing Dishes

A chef has collected data on the `satisfaction` level of his `n` dishes. Chef can cook any dish in 1 unit of time.

**Like-time coefficient** of a dish is defined as the time taken to cook that dish including previous dishes multiplied by its satisfaction level i.e. `time[i] * satisfaction[i]`.

Return *the maximum sum of **like-time coefficient** that the chef can obtain after dishes preparation*.

Dishes can be prepared in **any** order and the chef can discard some dishes to get this maximum value.

 

**Example 1:**

```
Input: satisfaction = [-1,-8,0,5,-9]
Output: 14
Explanation: After Removing the second and last dish, the maximum total like-time coefficient will be equal to (-1*1 + 0*2 + 5*3 = 14).
Each dish is prepared in one unit of time.
```

**Example 2:**

```
Input: satisfaction = [4,3,2]
Output: 20
Explanation: Dishes can be prepared in any order, (2*1 + 3*2 + 4*3 = 20)
```

**Example 3:**

```
Input: satisfaction = [-1,-4,-5]
Output: 0
Explanation: People do not like the dishes. No dish is prepared.
```

 

**Constraints:**

- `n == satisfaction.length`
- `1 <= n <= 500`
- `-1000 <= satisfaction[i] <= 1000`

#### My approach

1. Since dishes can be prepared in any order, we don’t need to keep the original list’s order.

2. sort the original list in ==descending== order.

3. iterate sorted list, from bigger number to smaller number.

4. there are two number that we need to keep track on: ==total amount== of all dishes that have been chosen, and the ==Like-time coefficient== that we should return as answer.

   > for instance, we chose dishes from `d0` to `di`, in ==descending== order.
   >
   > Total i = sum([d0, d1, .., di])
   >
   > Like-time i = sum([d0 * (i + 1), d1 * (i), .., di * 1])
   >
   > in next loop, we chose `di+1`, these two number should be updated like this:
   >
   > total i+1 = total i + di+1
   >
   > Like-time i+1 = sum([d0 * (i + 2), d1 * (i + 1), .., di * 2, di+1 * 1]) = Like-time i + total i+1

```python
def maxSatisfaction(self, satisfaction: List[int]) -> int:
    # iterate in reverse order
    satisfaction = sorted(satisfaction)[::-1]
    ans = total = 0
    for sat in satisfaction:
        total += sat
        if total > 0: ans += total
        else: break
    return ans
```

### 1807. Evaluate the Bracket Pairs of a String

You are given a string `s` that contains some bracket pairs, with each pair containing a **non-empty** key.

- For example, in the string `"(name)is(age)yearsold"`, there are **two** bracket pairs that contain the keys `"name"` and `"age"`.

You know the values of a wide range of keys. This is represented by a 2D string array `knowledge` where each `knowledge[i] = [keyi, valuei]` indicates that key `keyi` has a value of `valuei`.

You are tasked to evaluate **all** of the bracket pairs. When you evaluate a bracket pair that contains some key `keyi`, you will:

- Replace `keyi` and the bracket pair with the key's corresponding `valuei`.
- If you do not know the value of the key, you will replace `keyi` and the bracket pair with a question mark `"?"` (without the quotation marks).

Each key will appear at most once in your `knowledge`. There will not be any nested brackets in `s`.

Return *the resulting string after evaluating **all** of the bracket pairs.*

 

**Example 1:**

```
Input: s = "(name)is(age)yearsold", knowledge = [["name","bob"],["age","two"]]
Output: "bobistwoyearsold"
Explanation:
The key "name" has a value of "bob", so replace "(name)" with "bob".
The key "age" has a value of "two", so replace "(age)" with "two".
```

**Example 2:**

```
Input: s = "hi(name)", knowledge = [["a","b"]]
Output: "hi?"
Explanation: As you do not know the value of the key "name", replace "(name)" with "?".
```

**Example 3:**

```
Input: s = "(a)(a)(a)aaa", knowledge = [["a","yes"]]
Output: "yesyesyesaaa"
Explanation: The same key can appear multiple times.
The key "a" has a value of "yes", so replace all occurrences of "(a)" with "yes".
Notice that the "a"s not in a bracket pair are not evaluated.
```

 

**Constraints:**

- `1 <= s.length <= 105`
- `0 <= knowledge.length <= 105`
- `knowledge[i].length == 2`
- `1 <= keyi.length, valuei.length <= 10`
- `s` consists of lowercase English letters and round brackets `'('` and `')'`.
- Every open bracket `'('` in `s` will have a corresponding close bracket `')'`.
- The key in each bracket pair of `s` will be non-empty.
- There will not be any nested bracket pairs in `s`.
- `keyi` and `valuei` consist of lowercase English letters.
- Each `keyi` in `knowledge` is unique.

#### My approach

with re.findall and str.replace

```python
def evaluate(self, s: str, knowledge: List[List[str]]) -> str:
    knowledge = {i: j for i, j in knowledge}
    for key in set(re.findall('\([a-z]+\)', s)):
        if key[1:-1] in knowledge:
            s = s.replace(key, knowledge[key[1:-1]])
        else:
            s = s.replace(key, '?')
    return s
```

terrible time complexity.. 6700ms

```python
def evaluate(self, s: str, knowledge: List[List[str]]) -> str:
    knowledge = dict(knowledge)
    s = s.split('(')
    for idx, sub in enumerate(s):
        if ')' not in sub: continue
        sub = sub.split(')')
        s[idx] = (knowledge[sub[0]] if sub[0] in knowledge else '?') + sub[1]
    return ''.join(s)
```

Better.. 1300ms

### 423. Reconstruct Original Digits from English

Given a string `s` containing an out-of-order English representation of digits `0-9`, return *the digits in **ascending** order*.

 

**Example 1:**

```
Input: s = "owoztneoer"
Output: "012"
```

**Example 2:**

```
Input: s = "fviefuro"
Output: "45"
```

 

**Constraints:**

- `1 <= s.length <= 105`
- `s[i]` is one of the characters `["e","g","f","i","h","o","n","s","r","u","t","w","v","x","z"]`.
- `s` is **guaranteed** to be valid.

```python
def originalDigits(self, s: str) -> str:
    # TLE
    # cnt = collections.Counter(s)
    # digit = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    # def extract(cnt, idx):
    #     if all(i == 0 for i in cnt.values()): return (True, '')
    #     if idx == 10: return (False, '')

    #     d_cnt = collections.Counter(digit[idx])
    #     while True:
    #         for key, value in d_cnt.items():
    #             cnt[key] -= value
    #         if min(cnt.values()) >= 0:
    #             res = extract(cnt.copy(), idx + 1)
    #             if res[0]: return (True, str(idx) + res[1])
    #         else:
    #             for key, value in d_cnt.items():
    #                 cnt[key] += value
    #             res = extract(cnt.copy(), idx + 1)
    #             if res[0]: return (True, res[1])

    # return extract(cnt, 0)[1]

    """
    z in zero
    x in six
    w in two
    u in four
    g in eight
    h in three
    o in one
    f in five
    i in nine
    s in seven
    """
    ans = ''
    cnt = collections.Counter(s)
    digit = [('z', 'zero', 0), ('x', 'six', 6), ('w', 'two', 2), ('u', 'four', 4), ('g', 'eight', 8), ('h', 'three', 3), ('o', 'one', 1), ('f', 'five', 5), ('i', 'nine', 9), ('s', 'seven', 7)]
    digit_cnt = [0] * 10
    for ch, letter, idx in digit:
        temp = cnt[ch]
        for c in letter: cnt[c] -= temp
        digit_cnt[idx] = temp
    for idx, count in enumerate(digit_cnt):
        ans += str(idx) * count
    return ans
```

### 2034. Stock Price Fluctuation

You are given a stream of **records** about a particular stock. Each record contains a **timestamp** and the corresponding **price** of the stock at that timestamp.

Unfortunately due to the volatile nature of the stock market, the records do not come in order. Even worse, some records may be incorrect. Another record with the same timestamp may appear later in the stream **correcting** the price of the previous wrong record.

Design an algorithm that:

- **Updates** the price of the stock at a particular timestamp, **correcting** the price from any previous records at the timestamp.
- Finds the **latest price** of the stock based on the current records. The **latest price** is the price at the latest timestamp recorded.
- Finds the **maximum price** the stock has been based on the current records.
- Finds the **minimum price** the stock has been based on the current records.

Implement the `StockPrice` class:

- `StockPrice()` Initializes the object with no price records.
- `void update(int timestamp, int price)` Updates the `price` of the stock at the given `timestamp`.
- `int current()` Returns the **latest price** of the stock.
- `int maximum()` Returns the **maximum price** of the stock.
- `int minimum()` Returns the **minimum price** of the stock.

 

**Example 1:**

```
Input
["StockPrice", "update", "update", "current", "maximum", "update", "maximum", "update", "minimum"]
[[], [1, 10], [2, 5], [], [], [1, 3], [], [4, 2], []]
Output
[null, null, null, 5, 10, null, 5, null, 2]

Explanation
StockPrice stockPrice = new StockPrice();
stockPrice.update(1, 10); // Timestamps are [1] with corresponding prices [10].
stockPrice.update(2, 5);  // Timestamps are [1,2] with corresponding prices [10,5].
stockPrice.current();     // return 5, the latest timestamp is 2 with the price being 5.
stockPrice.maximum();     // return 10, the maximum price is 10 at timestamp 1.
stockPrice.update(1, 3);  // The previous timestamp 1 had the wrong price, so it is updated to 3.
                          // Timestamps are [1,2] with corresponding prices [3,5].
stockPrice.maximum();     // return 5, the maximum price is 5 after the correction.
stockPrice.update(4, 2);  // Timestamps are [1,2,4] with corresponding prices [3,5,2].
stockPrice.minimum();     // return 2, the minimum price is 2 at timestamp 4.
```

 

**Constraints:**

- `1 <= timestamp, price <= 10 ** 9`
- At most `10 ** 5` calls will be made **in total** to `update`, `current`, `maximum`, and `minimum`.
- `current`, `maximum`, and `minimum` will be called **only after** `update` has been called **at least once**.

#### My approach

use two heap queues, one tracks minimum price, the other tracks maximum price.

```python
import collections
import heapq

class StockPrice:

    def __init__(self):
        self.price = collections.defaultdict(int)
        self.latest = 0
        self.max_pq = []
        self.min_pq = []

    def update(self, timestamp: int, price: int) -> None:
        self.price[timestamp] = price
        self.latest = max(self.latest, timestamp)
        heapq.heappush(self.min_pq, (price, timestamp))
        heapq.heappush(self.max_pq, (-price, timestamp))

    def current(self) -> int:
        return self.price[self.latest]

    def maximum(self) -> int:
        while -self.max_pq[0][0] != self.price[self.max_pq[0][1]]:
            heapq.heappop(self.max_pq)
        return -self.max_pq[0][0]

    def minimum(self) -> int:
        while self.min_pq[0][0] != self.price[self.min_pq[0][1]]:
            heapq.heappop(self.min_pq)
        return self.min_pq[0][0]

if __name__ == '__main__':
    # Your StockPrice object will be instantiated and called as such:
    obj = StockPrice()
    obj.update(1, 3)
    print(obj.current())
    obj.update(3, 5)
    obj.update(1, 7)
    print(obj.maximum())
    print(obj.minimum())
```

### 1864. Minimum Number of Swaps to Make the Binary String Alternating

Given a binary string `s`, return *the **minimum** number of character swaps to make it **alternating**, or* `-1` *if it is impossible.*

The string is called **alternating** if ==no two adjacent characters are equal==. For example, the strings `"010"` and `"1010"` are alternating, while the string `"0100"` is not.

Any two characters may be swapped, even if they are **not adjacent**.

 

**Example 1:**

```
Input: s = "111000"
Output: 1
Explanation: Swap positions 1 and 4: "111000" -> "101010"
The string is now alternating.
```

**Example 2:**

```
Input: s = "010"
Output: 0
Explanation: The string is already alternating, no swaps are needed.
```

**Example 3:**

```
Input: s = "1110"
Output: -1
```

 

**Constraints:**

- `1 <= s.length <= 1000`
- `s[i]` is either `'0'` or `'1'`.

#### My approach

generate the expected format

1. count 0’s and 1’s amount in original string
2. if their amounts’ diff is greater than or equal to 2, then it is impossible to make it alternating, return -1
3. if diff is 1, then there should be an unique target format
4. if diff is 0, then there should be two different formats, one is started with ‘0’, ans the other is started with ‘1’.
5. compare target format ans original string, count the different chars, which are going to be swapped.

```python
def minSwaps(self, s: str) -> int:
    n = len(s)
    cnt = collections.Counter(s)
    diff = abs(cnt['0'] - cnt['1'])
    if diff > 1: return -1
    if diff == 1:
        if cnt['0'] > cnt['1']: template = [chr(ord('0') + i % 2) for i in range(n)]
        else: template = [chr(ord('1') - i % 2) for i in range(n)]
    else: # diff == 0
        template = [chr(ord('0') + i % 2) for i in range(n)]
    ans = sum(i != j for i, j in zip(s, template))
    return ans // 2 if diff == 1 else min(ans, n - ans) // 2
```

### 1377. Frog Position After T Seconds

Given an undirected tree consisting of `n` vertices numbered from `1` to `n`. A frog starts jumping from **vertex 1**. In one second, the frog jumps from its current vertex to another **unvisited** vertex if they are directly connected. The frog can not jump back to a visited vertex. In case the frog can jump to several vertices, it jumps randomly to one of them with the same probability. Otherwise, when the frog can not jump to any unvisited vertex, it jumps forever on the same vertex.

The edges of the undirected tree are given in the array `edges`, where `edges[i] = [ai, bi]` means that exists an edge connecting the vertices `ai` and `bi`.

*Return the probability that after `t` seconds the frog is on the vertex `target`.* Answers within `10 ** -5` of the actual answer will be accepted.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/12/21/frog1.jpg)

```
Input: n = 7, edges = [[1,2],[1,3],[1,7],[2,4],[2,6],[3,5]], t = 2, target = 4
Output: 0.16666666666666666 
Explanation: The figure above shows the given graph. The frog starts at vertex 1, jumping with 1/3 probability to the vertex 2 after second 1 and then jumping with 1/2 probability to vertex 4 after second 2. Thus the probability for the frog is on the vertex 4 after 2 seconds is 1/3 * 1/2 = 1/6 = 0.16666666666666666. 
```

**Example 2:**

**![img](https://assets.leetcode.com/uploads/2021/12/21/frog2.jpg)**

```
Input: n = 7, edges = [[1,2],[1,3],[1,7],[2,4],[2,6],[3,5]], t = 1, target = 7
Output: 0.3333333333333333
Explanation: The figure above shows the given graph. The frog starts at vertex 1, jumping with 1/3 = 0.3333333333333333 probability to the vertex 7 after second 1. 
```

 

**Constraints:**

- `1 <= n <= 100`
- `edges.length == n - 1`
- `edges[i].length == 2`
- `1 <= ai, bi <= n`
- `1 <= t <= 50`
- `1 <= target <= n`

```python
def frogPosition(self, n: int, edges: List[List[int]], t: int, target: int) -> float:
    """calculate the possibility that after t jumps the frog is on the target vertex

    Args:
        n (int): total amount of vertices
        edges (List[List[int]]): 2-d list represents all edges in tree
        t (int): limit t jumps
        target (int): specify the target vertex

    Returns:
        float: the possibility for frog to stay on target, return 0 if frog cannot reach target in t jumps or passed target in less than t jumps and target has no child
    """
    tree = collections.defaultdict(list)
    for i, j in edges:
        tree[i].append(j)
        tree[j].append(i)
    visited = set()
    curr = {1: 1} # frog is initially on vertex 1, and its possibility is 1/1 after 0 jump
    for _ in range(t):
        # frog reached target vertex in less than t jumps
        if target in curr:
            # there is still other vertex to jump, frog will not stuck on target
            if any(c not in visited for c in tree[target]): return 0
            # there is no vertex to jump, frog will stuck on target
            else: return 1 / curr[target]
        temp = dict()
        for node, possibility in curr.items():
            visited.add(node)
            nxt = [c for c in tree[node] if c not in visited]
            for n in nxt: temp[n] = possibility * len(nxt)
        curr = temp
    return 1 / curr[target] if target in curr else 0 # frog failed to reach target within t jumps
```
