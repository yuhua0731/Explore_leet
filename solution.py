#!/usr/bin/env python3
from typing import List
import csv
import time
import random
import collections
import math
import functools
import itertools
import heapq


class ListNode:
    # Definition for singly-linked list.
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def printList(self):
        temp = self
        while temp:
            print(temp.val, end='')
            temp = temp.next
        print('')


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

    def printNode(self):
        if self: print(self.val)
        if self.left: self.left.printNode()
        if self.right: self.right.printNode()


class fileHandler:
    def __init__(self):
        self.timestr = time.strftime("%Y%m%d%H%M%S")

    def writer(self):
        self.timestr = time.strftime("%Y%m%d%H%M%S")
        with open(self.timestr+'.csv', 'w', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['TimeDiff(us)', 'Cobid', 'Length', 'RTR', 'Data'])

    def take_writer(self):
        with open(self.timestr+'.csv', 'w', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow('hello')


class NumArray:
    # Your NumArray object will be instantiated and called as such:
    # obj = NumArray(nums)
    # param_1 = obj.sumRange(left,right)

    # TLE
    # sum_range = collections.defaultdict(list)
    # def __init__(self, nums: List[int]):
    #     n = len(nums)
    #     for i in range(n):
    #         for value in self.sum_range.values():
    #             value.append(value[-1] + nums[i])
    #         self.sum_range[i].append(nums[i])

    # def sumRange(self, left: int, right: int) -> int:
    #     return self.sum_range[left][right - left]

    part = [0]

    def __init__(self, nums: List[int]):
        # append additional element at the front of list
        # prevent from checking if index out of range
        for i in nums:
            self.part.append(self.part[-1] + i)

    def sumRange(self, left: int, right: int) -> int:
        # sum to right - sum to (left - 1)
        # left & right are inclusive
        return self.part[right + 1] - self.part[left]


class Solution:
    def __init__(self) -> None:
        super().__init__()

    def isIsomorphic(self, s: str, t: str) -> bool:
        mappings = dict()
        for i in range(len(s)):
            if s[i] in mappings:
                if mappings[s[i]] != t[i]:
                    return False
            else:
                if t[i] in mappings.values():
                    return False
                mappings[s[i]] = t[i]
        return True

    def numDecodingsWithStar(self, s: str) -> int:
        """
        dp[0] represents for ending with one digit number
        dp[1] represents for total decodings when last digit is '1'
        dp[2] represents for total decodings when last digit is '2'
        """
        mod = 10 ** 9 + 7
        dp, dp_new = [1, 0, 0], [0] * 3
        for c in s:
            if c == '*':
                dp_new[0] = 9 * dp[0] + 9 * dp[1] + 6 * dp[2]
                dp_new[1] = dp[0]
                dp_new[2] = dp[0]
            else:
                dp_new[0] = (c != '0') * dp[0] + dp[1] + (int(c) < 7) * dp[2]
                dp_new[1] = (c == '1') * dp[0]
                dp_new[2] = (c == '2') * dp[0]
            dp = [i % mod for i in dp_new]
        return dp[0]

    def lengthOfLIS(self, nums: List[int]) -> int:
        # DFS, failed TLE
        # n = len(nums)
        # global res
        # res = 0
        # def search_next(index, curr_max, len):
        #     global res
        #     if index <= n:
        #         for i in range(index, n):
        #             if nums[i] > curr_max:
        #                 search_next(i + 1, nums[i], len + 1)
        #     res = max(res, len)
        #     return
        # search_next(0, - 10 ** 4 - 1, 0)
        # return res

        # classic dp problem
        # dp[i] represents for LIS end with nums[i]
        # time complexity = O(n^2)
        # space complexity = O(n)
        # n = len(nums)
        # dp = [1] * n
        # for i in range(n):
        #     for j in range(i + 1, n):
        #         if nums[i] < nums[j]:
        #             dp[j] = max(dp[j], dp[i] + 1)
        # return max(dp)

        # optimized dp algorithm
        # time complexity = O(nlogn)
        # space complexity = O(n)
        # beats 80+% and 90+%
        tails = list()
        tails.append(nums[0])
        for i in range(1, len(nums)):
            if nums[i] > tails[-1]:
                tails.append(nums[i])
            else:
                left, right = 0, len(tails) - 1
                while left != right:
                    middle = (left + right) // 2
                    if tails[middle] < nums[i]:
                        left = middle + 1
                    else:
                        right = middle
                tails[left] = min(tails[left], nums[i])
        return len(tails)

    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        # forgive me, this is find subsequence
        # m, n = len(nums1), len(nums2)
        # def find_next(i, j):
        #     if i == m or j == n:
        #         return 0
        #     if nums1[i] == nums2[j]:
        #         return 1 + find_next(i + 1, j + 1)
        #     return max(find_next(i, j + 1), find_next(i + 1, j))
        # return find_next(0, 0)

        # find subarray
        m, n = len(nums1), len(nums2)
        dp = [[0] * (n + 1) for i in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = 1 + dp[i - 1][j - 1]
        return max(max(x) for x in dp)

    def findPeakElement(self, nums: List[int]) -> int:
        # O(logn) time
        # check middle element
        # if it is a peak element, return its index
        left, right = 0, len(nums) - 1
        while left != right:
            middle = (left + right) // 2
            if middle - 1 >= 0 and nums[middle - 1] >= nums[middle]:
                right = middle
            elif middle + 1 < len(nums) and nums[middle + 1] >= nums[middle]:
                left = middle + 1
            else:
                return middle
        return left

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        ret, curr_four = list(), list()
        nums.sort()
        n = len(nums)

        def find_next(index: int, sum: int) -> bool:
            # reached end of nums
            if len(curr_four) < 2:
                for i in range(index + 1, n):
                    if i > index + 1 and nums[i] == nums[i - 1]:
                        continue
                    curr_four.append(nums[i])
                    find_next(i, sum - nums[i])
                    curr_four.pop()
            else:
                l, r = index + 1, n - 1
                while l < r:
                    if nums[l] + nums[r] == sum:
                        ret.append(curr_four + [nums[l], nums[r]])
                        l += 1
                        r -= 1
                        while l < r and nums[l] == nums[l - 1]:
                            l += 1
                        while r > l and nums[r] == nums[r + 1]:
                            r -= 1
                    elif nums[l] + nums[r] < sum:
                        l += 1
                    else:
                        r -= 1

        find_next(-1, target)
        return ret

    def triangleNumber(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        global ret
        ret = 0

        # we pick the first edge
        # 从小到大轮询，后两条边需要做减法运算
        # 每当找到临界点，将第二条边更新，同时需要再次从最后一个index开始找临界点
        # for i in range(n):
        #     l, r = i + 1, n - 1
        #     while l < r:
        #         if nums[i] + nums[l] > nums[r]:
        #             print(i, l, r)
        #             ret += r - l
        #             l += 1
        #             r = n - 1
        #         else:
        #             r -= 1

        # we pick the last edge
        # 从大到小轮询，前两条边做加法运算
        # 每当找到临界点，将第一条边后移，第二条变无需重新找，可从当前index继续往前找
        for i in range(n - 1, 0, -1):
            l, r = 0, i - 1
            while l < r:
                if nums[l] + nums[r] > nums[i]:
                    ret += r - l
                    r -= 1
                else:
                    l += 1
        return ret

    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        # python pointer
        current = head
        next = None
        prev = None
        count = 0

        # Reverse first k nodes of the linked list
        while current and count < k:
            next, current.next, prev = current.next, prev, current
            current = next
            count += 1

        # next is now a pointer to (k+1)th node
        # recursively call for the list starting
        # from current. And make rest of the list as
        # next of first node
        if next is not None:
            head.next = self.reverseKGroup(next, k)

        # prev is new head of the input list
        return prev

    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        # accepted
        # global ancester
        # ancester = None
        # def find_index(node: TreeNode):
        #     global ancester
        #     ret = [False, False]
        #     if ancester:
        #         return ret
        #     ret[0] = True if node == p else False
        #     ret[1] = True if node == q else False
        #     if node and not all(i for i in ret):
        #         ret_left = find_index(node.left)
        #         ret_right = find_index(node.right)
        #         ret[0] |= ret_left[0] | ret_right[0]
        #         ret[1] |= ret_left[1] | ret_right[1]
        #     print(f'{node.val if node else None}: {ret}')
        #     if all(i for i in ret):
        #         ancester = node
        #     return ret

        # find_index(root)
        # return ancester

        # we realize that this is a binary search tree
        # which obey the rule: left.val < node.val < right.val
        def find_ancester(node: TreeNode, p_: int, q_: int):
            if node.val > p_ and node.val > q_:
                return find_ancester(node.left, p_, q_)
            if node.val < p_ and node.val < q_:
                return find_ancester(node.right, p_, q_)
            return node

        return find_ancester(root, p.val, q.val)

    class shuffle:
        def __init__(self, nums: List[int]):
            self.origin = nums.copy()
            self.sf = nums

        def reset(self) -> List[int]:
            """
            Resets the array to its original configuration and return it.
            """
            return self.origin

        # random sort key
        # sorted(self.nums, key = lambda _: random.random())
        def shuffle(self) -> List[int]:
            """
            Returns a random shuffling of the array.
            """
            random.shuffle(self.sf)
            return self.sf

    def pushDominoes(self, dominoes: str) -> str:
        for i in range(len(dominoes)):
            pass
        return dominoes

    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        # find middle node and put it to root
        # loop until all node are set
        def subtree(left: int, right: int) -> TreeNode:
            if right <= left:
                return None
            middle = left + (right - left) // 2
            ret = TreeNode(nums[middle])
            ret.left = subtree(left, middle)
            ret.right = subtree(middle + 1, right)
            return ret
        return subtree(0, len(nums))

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        # DFS
        # nums.sort()
        # ret = sum(nums[:3])
        # sums = 0
        # for i in range(len(nums)):
        #     if 3 * nums[i] >= ret and ret >= target:
        #         break
        #     sums = nums[i]
        #     for j in range(i + 1, len(nums)):
        #         if sums + 2 * nums[j] >= ret and ret >= target:
        #             break
        #         sums += nums[j]
        #         for k in range(j + 1, len(nums)):
        #             if sums + nums[k] >= ret and ret >= target:
        #                 break
        #             sums += nums[k]
        #             if abs(sums - target) < abs(ret - target):
        #                 ret = sums
        #                 if ret == target:
        #                     return ret
        #             sums -= nums[k]
        #         sums -= nums[j]
        #     sums -= nums[i]
        # return ret

        # more efficient way
        nums.sort()
        ret = sum(nums[:3])
        n = len(nums)
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            l, r = i + 1, n - 1
            ls, rs = nums[i] + nums[l] + \
                nums[l + 1], nums[i] + nums[r] + nums[r - 1]
            if ls > target:
                r = l + 1
            elif rs < target:
                l = r - 1
            while l < r:
                sums = nums[i] + nums[l] + nums[r]
                if abs(sums - target) < abs(ret - target):
                    ret = sums
                if sums == target:
                    return sums
                elif sums < target:
                    l += 1
                else:
                    r -= 1
        return ret

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # time complexity: O(n)
        visited = dict()
        for i in range(len(nums)):
            if target - nums[i] in visited:
                return [visited[target - nums[i]], i]
            elif nums[i] not in visited:
                visited[nums[i]] = i
        return [0, 0]

    def largestIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        surroundings = [[-1, 0], [0, -1]]
        # key: str formed by two indices
        # value: island index, can be used to access island list
        find_island = dict()
        island = list()
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    # extend island
                    temp = set()
                    for sur in surroundings:
                        newi, newj = i + sur[0], j + sur[1]
                        if f'{newi} {newj}' in find_island:
                            temp.add(find_island[f'{newi} {newj}'])
                    temp = list(temp)
                    if not temp:
                        # new island
                        find_island[f'{i} {j}'] = len(island)
                        island.append(1)
                    else:
                        # extend exist island
                        find_island[f'{i} {j}'] = temp[0]
                        island[temp[0]] += 1
                    # merge other island to temp[0]
                    for isl in temp[1:]:
                        for key, value in find_island.items():
                            if value == isl:
                                find_island[key] = temp[0]
                        island[temp[0]] += island[isl]
        ret = max(island) if island else 0
        surroundings = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for i in range(m):
            for j in range(n):
                if not grid[i][j]:
                    area = set()
                    for sur in surroundings:
                        newi, newj = i + sur[0], j + sur[1]
                        if f'{newi} {newj}' in find_island:
                            area.add(find_island[f'{newi} {newj}'])
                    ret = max(ret, sum(island[k] for k in area) + 1)
        return ret

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        # DFS
        # subset, not list, no order
        n = len(nums)
        ret = set()
        global subset
        subset = ''

        def extend(index: int):
            global subset
            if index == n:
                return
            for i in range(index, n):
                temp = len(f' {nums[i]}')
                subset += f' {nums[i]}'
                if subset not in ret:
                    ret.add(subset)
                    extend(i + 1)
                subset = subset[:(0 - temp)]
        ret.add(subset)
        nums.sort()
        extend(0)
        ret_list = list()
        for i in ret:
            ret_list.append(list(map(int, i.split(' ')[1:])))
        return ret_list

    def powerset(self, s):
        x = len(s)
        masks = [1 << i for i in range(x)]
        for i in range(1 << x):
            yield [ss for mask, ss in zip(masks, s) if i & mask]

    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        ret = []

        def find_next(node: TreeNode, path: List[int], curr_sum: int):
            temp = path.copy()
            temp.append(node.val)
            curr_sum += node.val
            if not node.left and not node.right:
                # leaf
                if curr_sum == targetSum:
                    ret.append(temp)
                    return
            if node.left:
                find_next(node.left, temp, curr_sum)
            if node.right:
                find_next(node.right, temp, curr_sum)
        if root:
            find_next(root, [], 0)
        return ret

    def stoneGame(self, piles: List[int]) -> bool:
        # dp[i, j] => [first player point, second player point]
        # represents for a game with piles[i, j + 1]
        n = len(piles)
        dp = [[[0] * 2 for i in range(n)] for i in range(n)]
        for i in range(n):
            dp[i][i][0] = piles[i]
        for step in range(1, n):
            for i in range(n):
                j = i + step
                if j >= n:
                    continue
                if dp[i + 1][j][1] + piles[i] > dp[i][j - 1][1] + piles[j]:
                    # take first
                    dp[i][j] = [dp[i + 1][j][1] + piles[i], dp[i + 1][j][0]]
                else:
                    # take last
                    dp[i][j] = [dp[i][j - 1][1] + piles[j], dp[i][j - 1][0]]
        # for i in range(n):
        #     print(dp[i])
        return dp[0][n - 1][0] > dp[0][n - 1][1]

    def matrixRankTransform(self, A):
        n, m = len(A), len(A[0])
        rank = [0] * (m + n)
        d = collections.defaultdict(list)
        for i in range(n):
            for j in range(m):
                d[A[i][j]].append([i, j])

        def find(i):
            if p[i] != i:
                p[i] = find(p[i])
            return p[i]

        for a in sorted(d):
            p = range(m + n)
            rank2 = rank[:]
            for i, j in d[a]:
                i, j = find(i), find(j + n)
                p[i] = j
                rank2[j] = max(rank2[i], rank2[j])
            for i, j in d[a]:
                rank[i] = rank[j + n] = A[i][j] = rank2[find(i)] + 1
        return A

    def matrixRankTransform(self, matrix: List[List[int]]) -> List[List[int]]:
        temp = collections.defaultdict(list)
        m, n = len(matrix), len(matrix[0])
        rank = [0] * (m + n)  # confused
        for i in range(m):
            for j in range(n):
                temp[matrix[i][j]].append([i, j])
        print(temp.items())

        # loop with sorted keys
        for i in sorted(temp):
            pass

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # beats 98% in time
        anagram = collections.defaultdict(list)
        for s in strs:
            anagram[''.join(sorted(s))].append(s)
        return list(anagram.values())

    def canReorderDoubled(self, arr: List[int]) -> bool:
        # count = collections.defaultdict(int)
        # for i in arr:
        #     count[i] += 1
        # arr.sort(key=functools.cmp_to_key(lambda x, y: 1 if abs(x) > abs(y) else -1))
        # for i in arr:
        #     if count[i] == 0:
        #         continue
        #     if count[i * 2] == 0:
        #         return False
        #     count[i] -= 1
        #     count[i * 2] -= 1
        # return sum(list(count.values())) == 0
        count = collections.Counter(arr)
        for key in sorted(count, key=abs):
            if count[key] > count[2 * key]:
                return False
            count[2 * key] -= count[key]
        return True

    def minWindow(self, s: str, t: str) -> str:
        # sliding windows
        left, right = 0, 1
        ret, exist = s, False
        count = collections.Counter(t)

        while right <= len(s):
            if s[right - 1] in count:
                count[s[right - 1]] -= 1
            if all(i <= 0 for i in count.values()):
                exist = True
                # contain all chars, start move left cursor
                while True:
                    if len(ret) > (right - left):
                        ret = s[left:right]
                    if s[left] in count:
                        count[s[left]] += 1
                    left += 1
                    if not all(i <= 0 for i in count.values()):
                        break
            right += 1
        return ret if exist else ""

    def goodNodes(self, root: TreeNode) -> int:
        global ret
        # DFS

        def find_path(node: TreeNode, path: List):
            global ret
            # node is a leaf
            if node.val >= (max(path) if path else node.val):
                ret += 1
            if node.left:
                path.append(node.val)
                find_path(node.left, path)
                path.pop()
            if node.right:
                path.append(node.val)
                find_path(node.right, path)
                path.pop()
        ret = 0
        find_path(root, [])
        return ret

    def numDecodings(self, s: str) -> int:
        """
        dp[0] represents for ending with one-digit number
        dp[1] represents for total decodings when last digit is '1'
        dp[2] represents for total decodings when last digit is '2'
        dp[3] represents for ending with two-digit number
        """
        dp, dp_new = [1, 0, 0, 0], [0] * 4
        for i in map(int, s):
            if i == 0:
                dp_new = [0, 0, 0, dp[1] + dp[2]]
            elif i == 1:
                dp_new = [dp[0] + dp[3], dp[0] + dp[3], 0, dp[1] + dp[2]]
            elif i == 2:
                dp_new = [dp[0] + dp[3], 0, dp[0] + dp[3], dp[1] + dp[2]]
            elif i in range(3, 7):
                dp_new = [dp[0] + dp[3], 0, 0, dp[1] + dp[2]]
            else:
                dp_new = [dp[0] + dp[3], 0, 0, dp[1]]
            if sum(dp) == 0:
                return 0
            dp = dp_new
        return dp[0] + dp[3]

    def maxProduct(self, root: TreeNode) -> int:
        sum_set = set()

        def sum_tree(node: TreeNode):
            if node:
                ret = node.val + sum_tree(node.left) + sum_tree(node.right)
                sum_set.add(ret)
                return ret
            else:
                return 0

        sum_node = sum_tree(root)
        MOD = 10 ** 9 + 7
        return max([(sum_node - i) * i for i in sum_set]) % MOD

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # A Sudoku board (partially filled) could be valid but is not necessarily solvable
        count = collections.defaultdict(set)
        # key: r1 - r9, c1 - c9, s1 - s9
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    continue
                if board[i][j] in count[f'r{i}'] or board[i][j] in count[f'c{j}'] or board[i][j] in count[f's{i // 3 * 3 + j // 3}']:
                    return False
                count[f'r{i}'].add(board[i][j])
                count[f'c{j}'].add(board[i][j])
                count[f's{i // 3 * 3 + j // 3}'].add(board[i][j])
        return True

    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        count = collections.defaultdict(set)
        # key: r1 - r9, c1 - c9, s1 - s9
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    continue
                count[f'r{i}'].add(board[i][j])
                count[f'c{j}'].add(board[i][j])
                count[f's{i // 3 * 3 + j // 3}'].add(board[i][j])

        # traverse from top-left to bottom-right
        def solve_next(i: int, j: int):
            while i < 9:
                # locate next item to be solved
                if board[i][j] != '.':
                    j += 1
                    if j == 9:
                        i += 1
                        j = 0
                else:
                    # DFS
                    for element in map(str, range(1, 10)):
                        if element in count[f'r{i}'] or element in count[f'c{j}'] or element in count[f's{i // 3 * 3 + j // 3}']:
                            continue
                        else:
                            board[i][j] = element
                            count[f'r{i}'].add(element)
                            count[f'c{j}'].add(element)
                            count[f's{i // 3 * 3 + j // 3}'].add(element)
                            if solve_next(i, j):
                                return True
                            board[i][j] = '.'
                            count[f'r{i}'].remove(element)
                            count[f'c{j}'].remove(element)
                            count[f's{i // 3 * 3 + j // 3}'].remove(element)
                    return False
            return True

        solve_next(0, 0)
        return board

    def rectangleArea(self, rectangles: List[List[int]]) -> int:
        # sort all x coordinates
        xs = sorted(
            set([x for x1, y1, x2, y2 in rectangles for x in [x1, x2]]))
        # ☆☆☆ refer to 12# in README.md ☆☆☆
        # sorted(set([x for x1, y1, x2, y2 in rectangles for x in [x1, x2]]))
        # is equivalent to the following codes:
        # temp = []
        # for x1, y1, x2, y2 in rectangles:
        #     for x in [x1, x2]:
        #         temp.append(x)
        # xs = sorted(set(temp))

        # form a dict which key is x_coordinate and value is the index
        x_i = {value: index for index, value in enumerate(xs)}

        L = []  # empty list
        for x1, y1, x2, y2 in rectangles:
            L.append([y1, x1, x2, 1])  # bottom edge
            L.append([y2, x1, x2, -1])  # top edge
            # these two lines form a rectangle
        L.sort()  # first key is y coordinate, second key is x1, and then x2

        cur_y = cur_x_sum = area = 0
        count = [0] * len(x_i)
        for y, x1, x2, signal in L:
            area += (y - cur_y) * cur_x_sum
            cur_y = y  # current y level
            for i in range(x_i[x1], x_i[x2]):
                count[i] += signal
                # signal = 1, start from here, add 1 to all x_range
                # signal = -1, end here, minus 1 from all x_range
            """
            ex. 
            input = [[0,0,2,2],[1,1,2,4]]
            count = [1, 1, 0] y = 0
                  ↓ [1, 2, 0] y = 1 there is an overlap cell | area += (y: 1 - curr_y: 0) * curr_x_sum: 2
                  ↓ [0, 1, 0] y = 2 area += (y: 2 - curr_y: 1) * curr_x_sum: 2
                  ↓ [0, 0, 0] y = 4 area += (y: 4 - curr_y: 2) * curr_x_sum: 1
            """
            cur_x_sum = sum(x2 - x1 if c else 0 for x1, x2,
                            c in zip(xs, xs[1:], count))
        return area % (10 ** 9 + 7)

    def findGCD(self, nums: List[int]) -> int:
        min_, max_ = min(nums), max(nums)
        while True:
            max_ = max_ % min_
            if max_ == 0:
                return min_
            min_, max_ = max_, min_

    def findDifferentBinaryString(self, nums: List[str]) -> str:
        n = len(nums)
        nums = sorted(set(nums))
        count = 0
        for i in nums:
            if int(i, 2) != count:
                ret = str(bin(count))[2:]
                while len(ret) < n:
                    ret = '0' + ret
                return ret
            count += 1
        ret = str(bin(count))[2:]
        while len(ret) < n:
            ret = '0' + ret
        return ret

    def minimizeTheDifference(self, mat: List[List[int]], target: int) -> int:
        # DFS?
        # m = len(mat)
        # for row in mat:
        #     row.sort()
        # init = sum([i[0] for i in mat])
        # if init >= target:
        #     return abs(init - target)
        # global diff
        # diff = abs(init - target)
        # def find_next(row: int, cur_sum: int):
        #     global diff
        #     # print(row, cur_sum, diff)
        #     if cur_sum - target >= diff or diff == 0:
        #         return
        #     if row == m - 1:
        #         # last row
        #         for i in mat[row]:
        #             diff = min(diff, abs((cur_sum + i) - target))
        #     else:
        #         for i in mat[row]:
        #             find_next(row + 1, cur_sum + i)
        # find_next(0, 0)
        # return diff
        nums = {0}
        for row in mat:
            nums = {x + i for x in row for i in nums}  # set
        return min(abs(target - x) for x in nums)

    def recoverArray(self, n: int, sums: List[int]) -> List[int]:
        total = sum(sums)
        n = len(sums)
        sum_arr = total / (n / 2)

    def findTarget(self, root: TreeNode, k: int) -> bool:
        global node_val
        node_val = set()

        def find_next(node: TreeNode):
            global node_val
            if node:
                if k - node.val in node_val:
                    return True
                else:
                    node_val.add(node.val)
                    return find_next(node.left) or find_next(node.right)
            return False
        return find_next(root)

    def complexNumberMultiply(self, num1: str, num2: str) -> str:
        num1_part = [int(i) for i in num1[:-1].split('+')]
        num2_part = [int(i) for i in num2[:-1].split('+')]
        ret_part = [num1_part[0] * num2_part[0] - num1_part[1] * num2_part[1],
                    num1_part[0] * num2_part[1] + num1_part[1] * num2_part[0]]
        return f'{str(ret_part[0])}+{str(ret_part[1])}i'

    def judgeSquareSum(self, c: int) -> bool:
        # 0 <= c <= 2 ** 31 - 1
        # def perfect_sqrt(i):
        #     return math.sqrt(i).is_integer()
        # return any(perfect_sqrt(c - a ** 2) for a in range(math.floor(math.sqrt(c / 2)) + 1))
        mid = math.floor(math.sqrt(c / 2))
        base = 0
        while base <= mid:
            if math.sqrt(c - base ** 2).is_integer():
                return True
            base += 1
        return False

    def isValidSerialization(self, preorder: str) -> bool:
        # node = preorder.split(',')
        # global index
        # index = 0
        # def traverse():
        #     global index
        #     if index >= len(node) or node[index] == '#':
        #         return
        #     else:
        #         # left child
        #         index += 1
        #         traverse()
        #         # right child
        #         index += 1
        #         traverse()
        # traverse()
        # return index == len(node) - 1

        # use stack
        node = []
        for i in preorder.split(','):
            while i == '#' and node and node[-1] == '#':
                # find two contiguous '#'
                # pop them out
                node.pop()
                # pop their parent node as well
                if not node:
                    return False
                node.pop()
            node.append(i)
        return len(node) == 1 and node[0] == '#'                
      
    def findLUSlength(self, strs: List[str]) -> int:
        # brute force, accepted
        strs.sort(key=functools.cmp_to_key(lambda x, y: 1 if len(x) < len(y) else -1))
        checked = set()
        def is_subsequence(a: str, b: str):
            index_a, index_b = 0, 0
            while index_a < len(a) and index_b < len(b):
                if a[index_a] == b[index_b]:
                    index_a += 1
                index_b += 1
            return index_a == len(a)

        for i in range(len(strs)):
            """
            - why do we only care about the whole word, rather than check all subsequences of it?
            - if the whole word is a common subsequence of another word, 
              then all subsequences of it are common subs of that word as well.
            """
            # check whole word
            if strs[i] not in checked:
                checked.add(strs[i])
                if all(not is_subsequence(strs[i], strs[j]) for j in range(len(strs)) if i != j):
                    return len(strs[i])
        return -1

    def minPatches(self, nums: List[int], n: int) -> int:
        nums.sort()
        cover, count, i = 1, 0, 0
        while n >= cover:
            if i < len(nums) and nums[i] <= cover:
                cover += nums[i]
                i += 1
            else:
                count += 1
                cover *= 2
        return count

    def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
        if not ops:
            return m * n
        # ops_x = sorted(ops)
        # ops_y = sorted(ops, key=functools.cmp_to_key(lambda a, b: 1 if a[1] > b[1] or (a[1] == b[1] and a[0] > b[0]) else -1))
        # return ops_x[0][0] * ops_y[0][1]
        return min(x for x, y in ops) * min(y for x, y in ops)
            
    def minimumDifference(self, nums: List[int], k: int) -> int:
        nums.sort()
        return min((nums[i + k - 1] - nums[i]) for i in range(len(nums) + 1 - k))

    def kthLargestNumber(self, nums: List[str], k: int) -> str:
        return str(tuple(sorted(map(int, nums), reverse=True))[k - 1])

    def minSessions(self, tasks: List[int], sessionTime: int) -> int:
        tasks.sort(reverse=True)
        session = list()
        global ret
        ret = len(tasks)
        def find_next(index: int):
            global ret
            if index == len(tasks) or any(se >= sum(tasks[index:]) for se in session):
                ret = min(len(session), ret)
                return
            i = tasks[index]
            for s in range(len(session)):
                if session[s] >= i:
                    session[s] -= i
                    find_next(index + 1)
                    session[s] += i
                s += 1
            session.append(sessionTime - i)
            find_next(index + 1)
            session.pop()
        find_next(0)
        return ret
            
    def numberOfUniqueGoodSubsequences(self, binary: str) -> int:
        """
        Q. 为什么保证unique?
        A. 在遍历i的时候，每次加上最后一位（"0"或"1"），都使得之前的集合增加了1位的长度，
        如果之前的集合是互相唯一的，那么在加上最后一位后仍为互相独立。同时，由于增加后集合里的元素
        最小长度为2，则需要补上1位长度的子序列。当i为"1"时，可以直接将其加入集合，因为元素允许以"1"
        作为起始，之后的遍历将会在这个"1"上增加长度。但是当i为"0"时，我们暂时不将其加入集合。因为
        元素不允许以"0"作为起始，如果加进去在后续遍历时则会产生非法子序列。这也是为什么在最后返回之前，
        我们需要将之前忽略的"0"加上去。

        Q. 为什么保证dp包含了所有可能的子序列？
        A. 不论子序列是由哪几位组成，其最后一位只有两种可能，即"0"或"1"。比如输入序列为"1001011",
        你可能会想，当遍历到最后一位时，前面的"(1001)011"子序列怎么被算进去？其实我们可以从最后一位往前倒推，
        这个子序列与"(1)0(0)1(0)1(1)"是相同的子序列。
        """
        # dp
        dp = [0, 0] # end with 0, end with 1
        MOD = 10 ** 9 + 7
        for i in binary:
            # add "1" as leading number in the further loop
            dp[int(i)] = (sum(dp) + int(i)) % MOD
        # add "0" to unique subs if "0" exists
        return (sum(dp) + ('0' in binary)) % MOD

    def findMin(self, nums: List[int]) -> int:
        # it's sorted, but with several retations
        # time complexity: O(logN)
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] >= nums[right]:
                left = mid + 1
            else:
                right = mid
        return nums[left]

    def arrayNesting(self, nums: List[int]) -> int:
        left = {i for i in range(len(nums))}
        visited = set()
        index = count = ret = 0
        while True:
            if index in visited:
                ret = max(count, ret)
                if not left:
                    break
                visited.clear()
                count = 0
                index = left.pop()
            visited.add(index)
            left.discard(index)
            index = nums[index]
            count += 1
        return ret

    def recoverArray(self, n: int, sums: List[int]) -> List[int]:
        def find(sums):
            if len(sums) == 1: return []
            counts = collections.Counter(sums)
            ele = sums[1] - sums[0]
            print(sums, ele)
            partA, partB = [], [] # A: all sums consist of ele | B: all sums not cal ele
            clear_flag = False
            for i in sums:
                # let's assume that i(sum of a subset) including ele
                # then i - ele will be another sum, and will be counted in counts
                # and temp do not include ele
                temp = i - ele
                if counts[i] > 0 and counts[temp] > 0:
                    counts[i] -= 1
                    counts[temp] -= 1
                    partA.append(i)
                    partB.append(temp)
                    # temp + ele = i
                    # if temp == 0, then ele is positive, and we can drop temp, take i to next loop
                    # if i == 0, then ele is negative, and we drop i, take temp to next level
                    # if temp == i == 0, then ele is 0, and partA & B are completely same, then both ways are acceptable
                    if temp == 0: clear_flag = True
            if clear_flag: return [ele] + find(partB)
            else: return [-ele] + find(partA)
        sums.sort()
        return find(sums)

    def minTimeToType(self, word: str) -> int:
        count = len(word)
        word = 'a' + word
        for i in range(1, len(word)):
            small, big = min(ord(word[i - 1]), ord(word[i])), max(ord(word[i - 1]), ord(word[i]))
            count += min(big - small, small - big + 26)
        return count

    def maxMatrixSum(self, matrix: List[List[int]]) -> int:
        # find a cell with negative value
        # search four neighbors, flip a set to see if their value increase
        # nega = set()
        # n = len(matrix)
        # neighbor = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        # for i in n:
        #     for j in n:
        #         if matrix[i][j] < 0:
        #             nega.add([i, j])
        # while True:
        #     if not nega:
        #         break
        #     for cell in nega:
        #         for neigh in neighbor:
        #             cell_ = [cell[0] + neigh[0], cell[1] + neigh[1]]
        #             if all(i >= 0 and i < n for i in cell_):
        #                 temp_sum = matrix[cell[0]][cell[1]] + matrix[cell_[0]][cell_[1]]
        #                 if temp_sum < 0:
        #                     pass
        # return sum([sum(i) for i in matrix])

        nega_ele = list()
        posi_min = 10 ** 5 + 1
        ans = 0
        for i in matrix:
            for j in i:
                if j >= 0:
                    if posi_min != 10 ** 5 + 1: ans += max(posi_min, j)
                    posi_min = min(posi_min, j)
                else:
                    nega_ele.append(j)
                    nega_ele.sort()
                    if len(nega_ele) == 3:
                        ans -= nega_ele.pop(0) + nega_ele.pop(0)
        if len(nega_ele) % 2 == 0:
            ans -= sum(nega_ele)
            ans += posi_min if posi_min != 10 ** 5 + 1 else 0
        else:
            ans += nega_ele[0] if posi_min == 10 ** 5 + 1 else abs(nega_ele[0] + posi_min)
        return ans
    
    def countPaths(self, n: int, roads: List[List[int]]) -> int:
        # dijkstra
        # 0 - (n - 1)
        # neighbors of all nodes
        # key: node
        # value: dict of its neighbors, and the time cost to travel
        neighbor = collections.defaultdict(dict)
        # shortest_path and its number of diff ways
        # key: destnation node
        # value: [shortest time, number of ways]
        shortest_path = dict()
        # form path & neighbor dict
        for e1, e2, cost in roads:
            neighbor[e1][e2] = cost
            neighbor[e2][e1] = cost

        # start from node 0, cost is 0, with only 1 way
        shortest_path[0] = [0, 1]
        # node, time, number of ways
        curr_state = [0, 0, 1]
        # loop until current shortest path belong to our target node
        visited = set() # prevent from revisiting a node, which will cause an endless loop
        while True:
            if curr_state[0] == n - 1:
                return curr_state[2] % (10 ** 9 + 7)
            visited.add(curr_state[0])
            for neigh, cost in neighbor[curr_state[0]].items():
                extend_time = curr_state[1] + cost
                if neigh not in shortest_path or shortest_path[neigh][0] > extend_time:
                    # this node has not visited beforem, or a shorter path is found
                    shortest_path[neigh] = [extend_time, curr_state[2]]
                elif shortest_path[neigh][0] == extend_time:
                    # same short path is found, add 1 to number of ways
                    shortest_path[neigh][1] += curr_state[2]
            # get overall shortest path, and update curr_state
            curr_state[1] = float('inf')
            for node, [cost, way] in shortest_path.items():
                if node not in visited and cost < curr_state[1]:
                    curr_state = [node, cost, way]

    def generateTrees(self, n: int) -> List[TreeNode]:
        # given a range (start, end), node val from start to end - 1
        # pick a root i, then its children can be formed from (start, i) & (i + 1, end)
        # form all sub trees of two children
        # combine two result lists
        def form_subtree(start: int, end: int) -> List[TreeNode]:
            if start >= end:
                return [None]
            ans = list()
            for root in range(start, end):
                left_child = form_subtree(start, root)
                right_child = form_subtree(root + 1, end)
                for left, right in list(itertools.product(left_child, right_child)):
                    root_node = TreeNode(root)
                    if left: root_node.left = left
                    if right: root_node.right = right
                    ans.append(root_node)
            return ans
        return form_subtree(1, n + 1)

    def numberOfCombinations(self, num: str) -> int:
        # positive, no leading zeros, non-decreasing
        # dp[i] is a dict
        # key: last number
        # value: count
        # TLE :(
        if num[0] == '0': return 0
        dp = list()
        # generate dp0 and append to dp list
        dp.append({0: 1})
        for i in range(1, len(num) + 1):
            dp0 = collections.defaultdict(int)
            for j in range(i):
                if num[j] == '0': continue
                last_number = int(num[j:i])
                for last, count in sorted(dp[j].items()):
                    if last_number >= last: dp0[last_number] += count
                    else: break
            dp.append(dp0)
        return sum(dp[-1].values())

    def outerTrees(self, trees: List[List[int]]) -> List[List[int]]:
        if len(trees) <= 1: return trees # only 1 tree
        def cross_product(A, B, C):
            return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

        ans = []
        trees.sort()
        for i in itertools.chain(range(len(trees)), reversed(range(len(trees)-1))):
            while len(ans) >= 2 and cross_product(ans[-2], ans[-1], trees[i]) < 0:
                ans.pop()
            ans.append(trees[i])
        ans.pop()
        res = []
        [res.append(x) for x in ans if x not in res]
        return res

    def orderlyQueue(self, s: str, k: int) -> str:
        # order first k letters in s, then find a position last should be inserted
        # if k == len(s): return ''.join(sorted(s))
        # first, last = sorted(s[:k]), s[k:]
        # for i in range(k):
        #     if ord(first[i]) > ord(last[0]): 
        #         return ''.join(first[:i]) + last + ''.join(first[i:])
        # return ''.join(first) + last
        if k == 1: # order will not be changed with any operation
            ans = s
            for i in range(len(s)):
                temp = s[i:] + s[:i]
                if temp < ans: ans = temp
            return ans
        return "".join(sorted(s))
        
    def countQuadruplets(self, nums: List[int]) -> int:
        # DFS
        global ans
        ans = 0

        def find_addition(index: int, s: int, count: int):
            global ans
            if count == 3:
                try:
                    while True:
                        temp = nums[index:].index(s)
                        ans += 1
                        index += temp + 1
                except:
                    return
            else:
                for i in range(index, len(nums)):
                    find_addition(i + 1, s + nums[i], count + 1)

        find_addition(0, 0, 0)
        return ans

    def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
        # sort attack in descending order, while defend in ascending order
        max_defend = ans = 0
        for _, defend in sorted(properties, key=lambda x: (-x[0], x[1])):
            if defend < max_defend: ans += 1
            else: max_defend = defend
        return ans

    def firstDayBeenInAllRooms(self, nextVisit: List[int]) -> int:
        # if you have been in room i an odd number of times (including the current visit)
        # on the next day you will visit the room specified by nextVisit[i]
        # if you have been in room i an even number of times (including the current visit)
        # on the next day you will visit room (i + 1) mod n

        # since nextVisit[i] is 0 ~ i, to visit i + 1, we have to visit i twice
        # dp[i] = dp[i - 1] + (dp[i - 1] - dp[nextVisit[i - 1]]) + 1 + 1
        #  
        # (first visit i - 1) + (second visit i - 1) - (first visit nextVisit[i - 1]) 
        # + (jump from i - 1 to nextVisit[i - 1]) + (visit i from i - 1)
        n = len(nextVisit)
        dp = [0] * n
        for i in range(1, n):
            dp[i] = (2 * dp[i - 1] - dp[nextVisit[i - 1]] + 2) % (10 ** 9 + 7)
        return dp[-1]

    def reverseList(self, head: ListNode) -> ListNode:
        # a  ->  b  ->  c
        pre_ = None
        curr_ = head
        # None     a  ->  b  ->  c
        # pre_   curr_
        while curr_:
            next_ = curr_.next
            # None     a   ->   b  ->  c
            # pre_   curr_     next_
            curr_.next = pre_
            # None <-  a        b  ->  c
            # pre_   curr_     next_
            pre_ = curr_
            # None <-  a        b  ->  c
            #      pre_&curr_  next_
            curr_ = next_
            # None <-  a        b  ->  c
            #         pre_  curr_&next_
        
        # None  <-   a   <-   b   <-   c       None
        #                             pre_   curr_&next_
        return pre_

    def gcdSort(self, nums: List[int]) -> bool:
        # if two nodes has the same root, then they are connected
        root = [i for i in range(max(nums) + 1)]
        
        # simple gcd function
        # def gcd(a: int, b: int) -> int:
        #     if a < b: a, b = b, a
        #     while b != 0:
        #         a %= b
        #         a, b = b, a
        #     return a

        def find(x) -> int:
            if root[x] != x: root[x] = find(root[x])
            return root[x]

        def union(x, y):
            # this will not stuck in an endless loop
            # find(x) was called first
            # ----- in find function, we update root[x] simultaneously -----!!!
            # find will exit recursion until root[x] == x
            # thus, root[find(x)] == find(x) is guaranteed
            root[find(x)] = find(y)

        def sieve(n: int) -> list:  # O(N*log(logN)) ~ O(N)
            spf = [i for i in range(n)]
            for i in range(2, n):
                if spf[i] != i: continue  # Skip if it's a not prime number
                for j in range(i * i, n, i):
                    if spf[j] > i:  # update to the smallest prime factor of j
                        spf[j] = i
            return spf
        spf = sieve(max(nums) + 1)

        def get_prime_factors(n: int) -> list:
            ans = list()
            while n > 1:
                ans.append(spf[n])
                n //= spf[n]
            return ans
        
        # use values as nodes
        for num in nums:
            for factor in get_prime_factors(num):
                # found a new edge
                union(factor, num)
        print(root)
        sorted_nums = sorted(nums)
        for snum, num in zip(sorted_nums, nums):
            if find(snum) != find(num):
                return False
        return True

        # use indices as nodes
        # TLE
        # for i in range(n):
        #     for j in range(i + 1, n):
        #         if gcd(nums[i], nums[j]) > 1:
        #             # found a new edge
        #             union(i, j)
        # nums_e = list(enumerate(nums))
        # tranform = list(enumerate(sorted(nums_e, key=lambda x: x[-1])))
        # for target_pos, (origin_pos, _) in tranform:
        #     # if target_pos and origin_pos are in the same graph for all nodes, return TRUE
        #     if find(target_pos) != find(origin_pos):
        #         return False
        # return True

    def findMiddleIndex(self, nums: List[int]) -> int:
        total = sum(nums)
        left_sum = 0
        for i in range(len(nums)):
            if total - left_sum - nums[i] == left_sum:
                return i
            left_sum += nums[i]
        return -1

    def findFarmland(self, land: List[List[int]]) -> List[List[int]]:
        # dp
        ans, count = dict(), 0
        m, n = len(land), len(land[0])
        for x in range(m):
            for y in range(n):
                if land[x][y] == 1:
                    # farmland
                    if x - 1 >= 0 and land[x - 1][y] != 0:
                        land[x][y] = land[x - 1][y]
                        ans[land[x][y]][2], ans[land[x][y]][3] = x, y
                    elif y - 1 >= 0 and land[x][y - 1] != 0:
                        land[x][y] = land[x][y - 1]
                        ans[land[x][y]][2], ans[land[x][y]][3] = x, y
                    else:
                        # new start of a farmland
                        count += 1
                        land[x][y] = count
                        ans[count] = [x, y, x, y]
        return ans.values()

    def orderOfLargestPlusSign(self, n: int, mines: List[List[int]]) -> int:
        max_length = [[[0] * 4 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                max_length[i][j][:] = [j + 1, n - j, i + 1, n - i]
        
        # group by rows
        mine_row = collections.defaultdict(list)
        mine_column = collections.defaultdict(list)
        for x, y in mines:
            mine_row[x].append(y)
            mine_column[y].append(x)
        for row, mine in mine_row.items():
            # row = 0
            # mine = [3, 6]
            mine.sort()
            mine = [-1] + mine + [n]
            for i in range(len(mine) - 1):
                left, right = mine[i], mine[i + 1]
                for column in range(max(left, 0), right):
                    max_length[row][column][:2] = [column - left, right - column]
        for column, mine in mine_column.items():
            # column = 0
            # mine = [3, 6]
            mine.sort()
            mine = [-1] + mine + [n]
            for i in range(len(mine) - 1):
                up, down = mine[i], mine[i + 1]
                for row in range(max(up, 0), down):
                    max_length[row][column][2:] = [row - up, down - row]
        # print(max_length)
        ans = 0
        for row in max_length:
            for cell in row:
                ans = max(ans, min(cell))
        return ans

    def reachableNodes(self, edges: List[List[int]], maxMoves: int, n: int) -> int:
        ans = 0
        inter = dict()
        neighbor = collections.defaultdict(list)
        for x, y, cnt in edges:
            neighbor[x].append(y)
            neighbor[y].append(x)
            inter[(x, y)] = cnt
            inter[(y, x)] = cnt
        visited = set() # element is nodes, for original nodes, we store '1'
        curr = [] # element is (step, node)
        heapq.heappush(curr, (0, 0))
        while curr:
            curr_node = heapq.heappop(curr) # (step, node)
            if curr_node[1] not in visited:
                visited.add(curr_node[1])
                ans += 1
                curr_neighbor = neighbor[curr_node[1]]
                for nei in curr_neighbor:
                    if nei not in visited:
                        # this neighbor is not visited, check if we can reach it
                        count = inter[(curr_node[1], nei)]
                        if curr_node[0] + count + 1 <= maxMoves:
                            # yes, we can reach this neighbor with curr_node[0] + count + 1 steps
                            heapq.heappush(curr, (curr_node[0] + count + 1, nei))
                            ans += count # all subdivided nodes should be added to the answer
                            inter[(curr_node[1], nei)] = 0 # we remove all sub-nodes from inter
                            inter[(nei, curr_node[1])] = 0
                        else:
                            # no, we cannot reach this neighbor
                            temp = maxMoves - curr_node[0] # how many steps remain
                            ans += temp
                            inter[(curr_node[1], nei)] -= temp # we remove those sub-nodes from inter
                            inter[(nei, curr_node[1])] -= temp
                    else:
                        # this neighbor is visited before, just add subdivided nodes
                        ans += min(inter[(curr_node[1], nei)], maxMoves - curr_node[0])
        return ans
        """
        inter is a dict
        key is a tuple of two nodes
        value is total subdivided nodes between them, after some sub-nodes were reached, we decrease this value

        neighbor is a defaultdict of list
        key is a node
        value is a list of all connected nodes

        visited is a set
        store all visited nodes

        curr is a heapq
        element is (min steps to reach node i, i)
        """