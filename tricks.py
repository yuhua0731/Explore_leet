#!/usr/bin/env python3
import collections
import functools
from typing import List


def majorityElement(nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    res = list()
    temp = dict()
    for i in nums:
        temp[i] = 1 if i not in temp else temp[i] + 1
    for k, v in temp.items():
        if v > len(nums) / 3:
            res.append(k)
    return res


def partitionLabels(S):
    last_index = [-1] * 26
    for i in range(len(S)):
        last_index[ord(S[i]) - ord('a')] = i
    # got last appear index of all chars
    start = 0
    end = 1
    res = list()
    new_sub = True
    while end <= len(S):
        for i in range(start, end):
            if last_index[ord(S[i]) - ord('a')] >= end:
                end = last_index[ord(S[i]) - ord('a')] + 1
                new_sub = False
                break
        if new_sub:
            res.append(end - start)
            start = end
            end += 1
        else:
            new_sub = True
    return res


def largestOverlap(img1, img2):
    """
    :type img1: List[List[int]]
    :type img2: List[List[int]]
    :rtype: int
    """
    l = len(img1)

    # get all ones in two matrices
    def one_cells(matrix):
        ones = []
        for x in range(l):
            for y in range(l):
                if matrix[x][y] == 1:
                    ones.append((x, y))
        return ones

    ones_1 = one_cells(img1)
    ones_2 = one_cells(img2)
    max_overlap = 0

    # a dict with pairs <vector, count>
    # to find the most count of same vector
    vector_count = collections.defaultdict(int)
    for (x1, y1) in ones_1:
        for (x2, y2) in ones_2:
            vector = (x2 - x1, y2 - y1)
            vector_count[vector] += 1
            max_overlap = max(max_overlap, vector_count[vector])

    return max_overlap


def crackSafe(n, k):
    # total possible combinations k^n
    total = k**n
    comb = set()
    res = []

    def dfs(starter):
        if len(comb) == total:
            return
        for digit in map(str, range(k)):
            curr = (starter + digit)[1:]
            if curr not in comb:
                comb.add(curr)
                dfs(curr)
                res.append(digit)

    starter = "0" * n
    comb.add(starter)
    dfs(starter)
    return "".join(res) + starter


def maxNumberOfFamilies(n, reservedSeats):
    """
    :type n: int
    :type reservedSeats: List[List[int]]
    :rtype: int
    """
    # assign reserved seats in row
    reserve, res = collections.defaultdict(int), 0
    for x, y in reservedSeats:
        if y > 1 and y < 10:
            reserve[x] |= 1 << (y - 2)
    for x in range(1, n + 1):
        res += 0 if all(reserve[x] & i for i in (15, 60, 240)
                        ) else 2 if reserve[x] == 0 else 1
    return res


def findTheDifference(s, t):
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    res = collections.defaultdict(int)
    for ch in s:
        res[ch] += 1
    for ch in t:
        res[ch] -= 1
    for k, v in res.items():
        if v != 0:
            return k
    return ""


def findKthPositive(arr, k):
    """
    :type arr: List[int]
    :type k: int
    :rtype: int
    """
    next_iter = 1
    for i in arr:
        while i > next_iter:
            k -= 1
            if k == 0:
                return next_iter
            next_iter += 1
        next_iter += 1
    return arr[len(arr) - 1] + k


def sumSubseqWidths(A):
    """
    891. Sum of Subsequence Widths
    :type A: List[int]
    :rtype: int
    """
    B = sorted(A)
    res = 0
    mod = 10**9 + 7
    for i in range(len(B)):
        res += B[i] * ((1 << i) - (1 << (len(B) - i - 1)))
    return res % mod


def findLength(A, B):
    """
    718. Maximum Length of Repeated Subarray
    :type A: List[int]
    :type B: List[int]
    :rtype: int
    """
    m = len(A)
    n = len(B)
    if m == 0 or n == 0:
        return 0
    res = [[0] * (n + 1)for i in range(m + 1)]
    """
    2-D array
    [[v] * n] * n, is a trap!
    >>> a = [[0] * 3] * 3
    >>> a
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    >>> a[0][0]=1
    >>> a
    [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
    """
    max_ = 0
    for i in range(m):
        for j in range(n):
            if A[i] == B[j]:
                res[i + 1][j + 1] = res[i][j] + 1
                max_ = max(max_, res[i + 1][j + 1])

    return max_


def minimumDeleteSum(s1: str, s2: str) -> int:
    # DP
    # dp[i][j] represents for minDeleteSum(s1[i], s2[j])
    # s1[i] represents for the first i chars in s1
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for i in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            # if i == 0 and j == 0:
            #     dp[i][j] = 0
            # elif i == 0 or j == 0:
            #     dp[i][j] = dp[i][j - 1] + \
            #         ord(s2[j - 1]) if i == 0 else dp[i - 1][j] + ord(s1[i - 1])
            # elif s1[i - 1] == s2[j - 1]:  # the last char is same, and then regardless
            #     dp[i][j] = dp[i - 1][j - 1]
            # else:
            #     dp[i][j] = min(dp[i][j - 1] + ord(s2[j - 1]),
            #                    dp[i - 1][j] + ord(s1[i - 1]))
            dp[i][j] = 0 if i == 0 and j == 0 else dp[i][j - 1] + ord(s2[j - 1]) if i == 0 else dp[i - 1][j] + ord(
                s1[i - 1]) if j == 0 else dp[i - 1][j - 1] if s1[i - 1] == s2[j - 1] else min(dp[i][j - 1] + ord(s2[j - 1]), dp[i - 1][j] + ord(s1[i - 1]))
            # damn one line >y<
    return dp[m][n]


def mctFromLeafValues(arr):
    """
    1130. Minimum Cost Tree From Leaf Values
    :type arr: List[int]
    :rtype: int
    """
    # upvote for lee btw
    res = 0
    while len(arr) > 1:
        index = arr.index(min(arr))
        res += min(arr[index - 1: index] +
                   arr[index + 1: index + 2]) * arr.pop(index)
    return res


def maxSumAfterPartitioning(arr, k):
    """
    :type arr: List[int]
    :type k: int
    :rtype: int
    """
    dp = [0] * len(arr)
    dp[0] = arr[0]
    for i in range(1, len(arr)):
        max_last = arr[i]
        for ki in range(k):
            if i - ki < 0:
                break
            max_last = max(max_last, arr[i - ki])
            # i - ki - 1 can be -1, which should be treat as 0
            dp[i] = max(dp[i], (dp[i - ki - 1] if i - ki -
                                1 >= 0 else 0) + (ki + 1) * max_last)
    return dp[len(arr) - 1]


def stoneGame(piles):
    """
    :type piles: List[int]
    :rtype: bool
    """
    dp = [0] * len(piles)
    # sorry


def minFallingPathSum(A):
    """
    931. Minimum Falling Path Sum
    :type A: List[List[int]]
    :rtype: int
    """
    dim = len(A)
    dp = [[0] * dim for i in range(dim)]
    for i in range(dim):
        dp[dim - 1][i] = A[dim - 1][i]
    for i in range(dim - 2, -1, -1):
        for j in range(dim):
            # at most three conditions: fall to left, fall to right, fall to direct below
            dp[i][j] = A[i][j] + min(dp[i + 1][j], dp[i + 1][j + 1] if j + 1 < dim else float(
                "inf"), dp[i + 1][j - 1] if j - 1 >= 0 else float("inf"))
    return min(dp[0][j] for j in range(dim))


def mincostTickets(days, costs):
    """
    983. Minimum Cost For Tickets
    :type days: List[int]
    :type costs: List[int]
    :rtype: int
    """
    # DP
    last_day = days[len(days) - 1]
    dp = [0] * (last_day + 1)
    for i in range(1, last_day + 1):
        dp[i] = min(dp[i - 1] + costs[0], costs[1] + (dp[i - 7] if i - 7 >= 0 else 0),
                    costs[2] + (dp[i - 30] if i - 30 >= 0 else 0)) if i in days else dp[i - 1]
    return dp[last_day]


def nthPersonGetsNthSeat(n: int) -> float:
    # 1227. Airplane Seat Assignment Probability
    dp = [0] * (n + 1)
    if n > 0:
        dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] / 2 + (1 - dp[i - 1]) / 2
    return dp[n]


def largestNumber(nums):
    """
    179. Largest Number
    :type nums: List[int]
    :rtype: str
    """
    def cmp_nums(num1, num2):
        return -1 if num1 + num2 > num2 + num1 else 1  # in descending order
    res = "".join(sorted(map(str, nums), key=functools.cmp_to_key(cmp_nums)))
    return str(int(res))  # remove redundant 0 ahead


def calcEquation(equations, values, queries):
    """
    :type equations: List[List[str]]
    :type values: List[float]
    :type queries: List[List[str]]
    :rtype: List[float]
    """
    graph = dict()
    visited = set()

    # build a directed weight graph
    def buildGraph(equation, value):
        if equation[0] not in graph:
            graph[equation[0]] = dict()
        if equation[1] not in graph:
            graph[equation[1]] = dict()
        graph[equation[1]][equation[0]] = 1 / value
        graph[equation[0]][equation[1]] = value

    def findTarget(current, target):
        visited.add(current)
        if current == target:
            return 1
        for k, v in graph[current].items():
            if k not in visited:
                temp = findTarget(k, target)
                if temp != None:  # target found
                    return v * temp
        return None

    def findPath(x, y):  # DFS
        if x not in graph or y not in graph:
            return -1
        visited.clear()
        return findTarget(x, y)

    for i in range(len(values)):
        buildGraph(equations[i], values[i])

    return [findPath(x, y) if findPath(x, y) != None else -1 for (x, y) in queries]


def findPoisonedDuration(timeSeries, duration):
    """
    Teemo Attacking
    :type timeSeries: List[int]
    :type duration: int
    :rtype: int
    """
    poisonEndTime = -1
    durationCount = 0
    for i in timeSeries:
        durationCount += duration if i > poisonEndTime else i + duration - 1 - poisonEndTime
        poisonEndTime = i + duration - 1
    return durationCount


def numSubarrayProductLessThanK(nums, k):
    """
    Subarray Product Less Than K
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    # sliding windows
    left = right = res = 0
    dim = len(nums)
    currProd = 1
    while right < dim:
        currProd *= nums[right]
        while currProd >= k and left < dim:
            res += right - left
            currProd /= nums[left]
            left += 1
        right += 1
    res += (right - left) * (right - left + 1) // 2
    return max(res, 0)


def rob(nums):
    """
    House Robber
    :type nums: List[int]
    :rtype: int
    """
    dp1 = dp2 = 0
    for i in nums:
        dp1, dp2 = dp2, max(dp1 + i, dp2)
    return dp2


def insert(intervals, newInterval):
    """
    Insert Interval
    :type intervals: List[List[int]]
    :type newInterval: List[int]
    :rtype: List[List[int]]
    """
    res = []
    for start, end in intervals:
        if start > newInterval[1]:
            res.append([start, end])
        elif end < newInterval[0]:
            res.append([start, end])
        else:
            newInterval = [min(newInterval[0], start),
                           max(newInterval[1], end)]
    res.append(newInterval)
    return sorted(res)


def wordBreak(s, wordDict):
    """
    Word Break
    :type s: str
    :type wordDict: List[str]
    :rtype: bool
    """
    # DFS try TLE
    def findNextSegment(str):
        for i in range(len(str)):
            if str[:i + 1] in wordDict:
                if i + 1 == len(str) or findNextSegment(str[i + 1:]):
                    return True
        return False

    # DP try
    dim = len(s)
    res = [False] * (dim + 1)
    res[0] = True
    for i in range(dim):
        for j in range(i + 1):
            if res[j] and s[j: i + 1] in wordDict:
                res[i + 1] = True
    return res[dim]


def minOperations(logs):
    """
    1598. Crawler Log Folder
    :type logs: List[str]
    :rtype: int
    """
    res = 0
    for s in logs:
        res += 0 if s == "./" or (s == "../" and res ==
                                  0) else -1 if s == "../" and res > 0 else 1
    return res


def minOperationsMaxProfit(customers, boardingCost, runningCost):
    """
    1599. Maximum Profit of Operating a Centennial Wheel
    :type customers: List[int]
    :type boardingCost: int
    :type runningCost: int
    :rtype: int
    """
    max_profit = max_rotation = curr_wait = curr_profit = 0
    for i in range(len(customers)):
        curr_wait += customers[i]
        curr_profit += min(curr_wait, 4) * boardingCost - runningCost
        curr_wait -= min(curr_wait, 4)
        if max_profit < curr_profit:
            max_profit = curr_profit
            max_rotation = i + 1
    rotate = len(customers)
    while curr_wait > 0:
        curr_profit += min(curr_wait, 4) * boardingCost - runningCost
        curr_wait -= min(curr_wait, 4)
        if max_profit < curr_profit:
            max_profit = curr_profit
            max_rotation = rotate + 1
            rotate += 1
    return -1 if max_rotation == 0 else max_rotation


def firstMissingPositive(nums):
    """
    First Missing Positive
    :type nums: List[int]
    :rtype: int
    """
    nums = sorted(nums)
    curr = 1
    for i in nums:
        if i > curr:
            return curr
        if i == curr:
            curr += 1
    return curr


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def insertIntoBST(root: TreeNode, val: int) -> TreeNode:
    def searchNext(node: TreeNode, val: int) -> TreeNode:
        if node == None:
            return TreeNode(val, None, None)
        if node.val > val:
            node.left = searchNext(node.left, val)
        else:
            node.right = searchNext(node.right, val)
        return node
    return searchNext(root, val)


def bitwiseComplement(N: int) -> int:
    multi, res = 1, 0
    if N == 0:
        return 1
    while N > 0:
        temp = 1 - N % 2
        N //= 2
        res += multi * temp
        multi *= 2
    return res


def removeCoveredIntervals(intervals):
    """
    :type intervals: List[List[int]]
    :rtype: int
    """
    dim, res = len(intervals), 0
    intervals.sort(key=functools.cmp_to_key(
        lambda x, y: 1 if x[0] > y[0] or (x[0] == y[0] and x[1] < y[1]) else -1))
    remove = [False] * dim
    for i in range(dim):
        if remove[i]:
            continue
        for j in range(i + 1, dim):
            if remove[j]:
                continue
            if intervals[i][1] < intervals[j][0]:
                break
            if intervals[i][0] <= intervals[j][0] and intervals[i][1] >= intervals[j][1]:
                remove[j] = True
                res += 1
    return dim - res


def findPairs(nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    nums, pairs = sorted(nums), set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + k == nums[j]:
                if (nums[i], nums[j]) not in pairs:
                    pairs.add((nums[i], nums[j]))
                    break
            elif nums[i] + k < nums[j]:
                break
    return len(pairs)


def findMinArrowShots(points):
    """
    Minimum Number of Arrows to Burst Balloons
    :type points: List[List[int]]
    :rtype: int
    """
    points.sort(key=functools.cmp_to_key(
        lambda x, y: 1 if x[1] > y[1] else -1))
    index = count = 0
    while index < len(points):
        shot, count = points[index][1], count + 1
        while index < len(points) and points[index][0] <= shot:
            index += 1
    return count


def removeDuplicateLetters(s: str) -> str:
    # remove duplicate letters
    # point is: the result should be the first in alphabetical order
    index, res, result, last = dict(), ["!"] * len(s), "", -1
    for i in range(len(s)):
        if s[i] not in index:
            index[s[i]] = list()
        index[s[i]].append(i)
    for ch in range(ord('a'), ord('z') + 1):
        if chr(ch) in index.keys():
            for i in index[chr(ch)]:
                if i > last:  # find best place
                    res[i] = chr(ch)
                    last = i
                    break
                if i == index[chr(ch)][-1]:
                    res[index[chr(ch)][-1]] = chr(ch)
    for ch in res:
        result += ch if ch != "!" else ""
    return result


def simplifyPath(path: str) -> str:
    # simplify a path
    # any multiple consecutive slashes (i.e. '//') are treated as a single slash
    cmd, res = path.split('/'), ''
    pathlist = list()
    for i in cmd:
        if i == '' or i == '.':
            continue
        elif i == '..':
            if len(pathlist) > 0:
                pathlist.pop()
        else:
            pathlist.append(i)
    if len(pathlist) == 0:
        return '/'
    for i in pathlist:
        res += '/' + i
    return res


def matrixScore(A) -> int:
    # step 1: flip each row, setting most significant bit to 1
    # step 2: flip columns from 2nd column, making more 1's in that column
    m, n = len(A), len(A[0])
    def flip_row(row: int):
        for i in range(n):
            A[row][i] = 1 - A[row][i]
    
    def flip_column(col: int):
        for i in range(m):
            A[i][col] = 1 - A[i][col]

    def count_ones(col: int):
        res = 0
        for i in range(m):
            res += A[i][col]
        return res
        
    for i in range(m):
        if A[i][0] == 0:
            flip_row(i)
    
    for j in range(1, n):
        if count_ones(j) < m / 2:
            flip_column(j)
    
    sum = 0
    for i in A:
        prod = 2 ** (n - 1)
        for ele in i:
            sum += ele * prod
            prod //= 2
    return sum


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def hasCycle(head: ListNode) -> bool:
    # O(n^2)
    # visited = set()
    # while head != None:
    #     if head in visited:
    #         return True
    #     visited.add(head)
    #     head = head.next
    # return False

    # O(1) space complexity version
    if not head:
        return False
    # slow == fast at initial, cannot compare them before first movement
    slow = fast = head 
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if fast and slow.val == fast.val:
            return True
    return False


def findLHS(nums) -> int:
    # step 1: sort array
    # step 2: iterate array, record longest length
    nums, curr_min = sorted(nums), - 10 ** 9 - 2
    longest_len = curr_min_len = curr_max_len = 0
    for i in nums:
        if i == curr_min:
            curr_min_len += 1
        elif i == curr_min + 1:
            curr_max_len += 1
        else:
            # update to a new subsequence
            if curr_max_len > 0:
                longest_len = max(longest_len, curr_min_len + curr_max_len)
            # 1, 2, 3
            if curr_max_len > 0 and i == curr_min + 2:
                curr_min, curr_min_len, curr_max_len = curr_min + 1, curr_max_len, 1
            # 1, 2, 4, 5
            else:
                curr_min, curr_min_len, curr_max_len = i, 1, 0
    return max(longest_len, curr_min_len + curr_max_len) if curr_max_len > 0 else longest_len


def minOperations(n: int) -> int:
    # 1, 3, 5, 7...
    # if n % 2 == 0:
    #     # even, 1 + 3 + ...
    #     # return (n // 2 * 2 - 1 + 1) * (n // 2) // 2
    #     return (n ** 2) // 4
    # else:
    #     # odd, 2 + 4 + ...
    #     # return (n // 2 * 2 + 1) * (n // 2) // 2
    #     return (n + 1) * (n - 1) // 4
    return (n ** 2) // 4 if n % 2 == 0 else (n + 1) * (n - 1) // 4


def findingUsersActiveMinutes(logs: List[List[int]], k: int) -> List[int]:
    count = dict()
    res = [0] * k
    for action in logs:
        if action[0] not in count:
            count[action[0]] = set()
        if action[1] not in count[action[0]]:
            count[action[0]].add(action[1])
    for key, value in count.items():
        res[len(value) - 1] += 1
    return res
            

def minAbsoluteSumDiff(nums1: List[int], nums2: List[int]) -> int:
    total_sum, max_diff, index = 0, 0, -1
    for i in len(nums1):
        curr_diff = abs(nums1[i] - nums2[i])
        total_sum += curr_diff
        if max_diff < curr_diff:
            max_diff = curr_diff
            index = i
    if total_sum == 0:
        return total_sum

# pending
def numDecodings(s: str) -> int:
    mod = 10 ** 9 + 7
    n = len(s)
    dp = [0] * (n + 1) # dp[i] represents for total decodings of s[0:i]
    dp[0] = 1
    dp[1] = 1 if s[0] != '*' else 9
    for i in range(2, n + 1):
        if s[i - 1] == '*':
            dp[i] = (dp[i] + dp[i - 1] * 9) % mod
            if s[i - 2] == '1':
                dp[i] = (dp[i] + dp[i - 1] * 9) % mod
            elif s[i - 2] == '2':
                dp[i] = (dp[i] + dp[i - 1] * 6) % mod
            elif s[i - 2] == '*':
                dp[i] = (dp[i] + dp[i - 1] * 17) % mod
        elif s[i - 2 : i] == '*0':
            dp[i] = (dp[i] + dp[i - 2] * 2) % mod
        elif s[i - 2 : i] in ['10', '20']:
            dp[i] = (dp[i] + dp[i - 2]) % mod
        else:
            dp[i] = (dp[i] + dp[i - 1]) % mod
            if s[i - 2] == '1':
                dp[i] = (dp[i] + dp[i - 2]) % mod
            elif s[i - 2] == '2' and int(s[i - 1]) < 7:
                dp[i] = (dp[i] + dp[i - 2]) % mod
            elif s[i - 2] == '*':
                if int(s[i - 1]) < 7:
                    dp[i] = (dp[i] + dp[i - 2] * 2) % mod
                else:
                    dp[i] = (dp[i] + dp[i - 2]) % mod
    print(dp)
    return dp[n]


def funnyTricks():
    """
    python's magic
    """
    test = collections.defaultdict(int)
    print(test[1])
    # got KeyError!
    # test1 = dict()
    # print(test1[1])

    # map(func, arr), convert each element in arr by func
    arr = [1, 2, 3, 4]

    def calculateSquare(n):
        return n*n
    square = map(calculateSquare, arr)
    print(square)
    print(sorted(set(square)))
    square = map(str, arr)
    print(square)
    print(set(square))  # unsorted set, random

    res = []
    res.append("1")
    res.append("2")
    print("".join(res))

    # customer compare function in sort
    nums = [28, 50, 17, 12, 121]
    nums.sort(key=functools.cmp_to_key(
        lambda x, y: 1 if str(x) + str(y) < str(y) + str(x) else -1))
    print(nums)

    # remove duplicate items in a list
    nums = [0, 0, 1, 9, 9, 0]
    print(list(dict.fromkeys(nums)))

    # r prefix to a string
    print('r prefix to \'\\n\' is a backslash followed by letter n: ' + r'\n')
    print('without the r it will be a newline: ' + '\n')


def main():
    # print(majorityElement([1, 2, 1]))
    # print(partitionLabels("ababcbacadefegdehijhklij"))
    # print([1] * 5)
    # print(crackSafe(2, 2))
    # print(maxNumberOfFamilies(3, [[1,2],[1,3],[1,8],[2,6],[3,1],[3,10]]))
    # print(findTheDifference("asdf", "asdfg"))
    # print(findKthPositive([2,3,4,7,11], 5))
    # print(sumSubseqWidths([2,1,3]))
    # print(findLength([1, 0, 0, 0, 1], [1, 0, 0, 1, 1]))
    # print(minimumDeleteSum("sea", "eat"))
    # print(mctFromLeafValues([6, 2, 4]))
    # print(maxSumAfterPartitioning([1, 15, 7, 9, 2, 5, 10], 3))
    # print(minFallingPathSum([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    # print(mincostTickets([1, 4, 6, 7, 8, 20], [7, 2, 15]))
    # print(nthPersonGetsNthSeat(3))
    # print(largestNumber([2, 10, 30, 8]))
    # print(calcEquation([["a", "b"], ["b", "c"]], [2.0, 3.0], [
    #       ["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"]]))
    # print(calcEquation([["a","b"],["c","d"]], [1.0,1.0], [["a","c"],["b","d"],["b","a"],["d","c"]]))
    # print(findPoisonedDuration([0, 1, 2, 7, 10], 2))
    # print(numSubarrayProductLessThanK([10, 9, 10, 4, 3, 8, 3, 3, 6, 2, 10, 10, 9, 3], 19))
    # print(rob([2, 7, 9, 3, 1]))
    # print(insert([[1, 3], [6, 9]], [2, 5]))
    # print(wordBreak("leetcode", ["leet", "code"]))
    # print(minOperations(["d1/", "d2/", "../", "d21/", "./"]))
    # print(minOperationsMaxProfit([0,43,37,9,23,35,18,7,45,3,8,24,1,6,37,2,38,15,1,14,39,27,4,25,27,33,43,8,44,30,38,40,20,5,17,27,43,11,6,2,30,49,30,25,32,3,18,23,45,43,30,14,41,17,42,42,44,38,18,26,32,48,37,5,37,21,2,9,48,48,40,45,25,30,49,41,4,48,40,29,23,17,7,5,44,23,43,9,35,26,44,3,26,16,31,11,9,4,28,49,43,39,9,39,37,7,6,7,16,1,30,2,4,43,23,16,39,5,30,23,39,29,31,26,35,15,5,11,45,44,45,43,4,24,40,7,36,10,10,18,6,20,13,11,20,3,32,49,34,41,13,11,3,13,0,13,44,48,43,23,12,23,2]
    # , 43
    # , 54))
    # print(firstMissingPositive([7, 8, 9, 11, 12]))
    # print(firstMissingPositive([3, 4, -1, 1]))
    # print(insertIntoBST(TreeNode(5), 8))
    # print(bitwiseComplement(7))
    # print(removeCoveredIntervals([[34335,39239],[15875,91969],[29673,66453],[53548,69161],[40618,93111]]))
    # print(findPairs([7, 0, 1, 3, 5, 0], 0))
    # print(findMinArrowShots([[10, 16], [2, 6], [1, 6], [7, 12]]))
    # print(removeDuplicateLetters("cbacdcbc"))
    # print(simplifyPath("/a///./b//../../c"))
    # print(matrixScore([[0,0,1,1],[1,0,1,0],[1,1,0,0]]))
    # print(findLHS([2, 4, 3, 4]))
    # print(minOperations(6))
    # print(findingUsersActiveMinutes([[0, 5], [1, 2], [1, 5], [0, 3], [0, 5]], 5))
    print(numDecodings('**'))

if __name__ == "__main__":
    main()
