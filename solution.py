#!/usr/bin/env python3
from typing import List
import csv
import time
import random
import collections
from collections import Counter
import math
import functools
import itertools
from itertools import chain, product
import heapq

from trie import Trie
from union_find import UnionFind
import re
import bisect


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
        if self:
            print(self.val)
        if self.left:
            self.left.printNode()
        if self.right:
            self.right.printNode()


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


class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


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
        strs.sort(key=functools.cmp_to_key(
            lambda x, y: 1 if len(x) < len(y) else -1))
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
        dp = [0, 0]  # end with 0, end with 1
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
            if len(sums) == 1:
                return []
            counts = collections.Counter(sums)
            ele = sums[1] - sums[0]
            # print(sums, ele)
            partA, partB = [], []  # A: all sums consist of ele | B: all sums not cal ele
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
                    if temp == 0:
                        clear_flag = True
            if clear_flag:
                return [ele] + find(partB)
            else:
                return [-ele] + find(partA)
        sums.sort()
        return find(sums)

    def minTimeToType(self, word: str) -> int:
        count = len(word)
        word = 'a' + word
        for i in range(1, len(word)):
            small, big = min(ord(word[i - 1]), ord(word[i])
                             ), max(ord(word[i - 1]), ord(word[i]))
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
                    if posi_min != 10 ** 5 + 1:
                        ans += max(posi_min, j)
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
            ans += nega_ele[0] if posi_min == 10 ** 5 + \
                1 else abs(nega_ele[0] + posi_min)
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
        visited = set()  # prevent from revisiting a node, which will cause an endless loop
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
                    if left:
                        root_node.left = left
                    if right:
                        root_node.right = right
                    ans.append(root_node)
            return ans
        return form_subtree(1, n + 1)

    def numberOfCombinations(self, num: str) -> int:
        # positive, no leading zeros, non-decreasing
        # dp[i] is a dict
        # key: last number
        # value: count
        # TLE :(
        if num[0] == '0':
            return 0
        dp = list()
        # generate dp0 and append to dp list
        dp.append({0: 1})
        for i in range(1, len(num) + 1):
            dp0 = collections.defaultdict(int)
            for j in range(i):
                if num[j] == '0':
                    continue
                last_number = int(num[j:i])
                for last, count in sorted(dp[j].items()):
                    if last_number >= last:
                        dp0[last_number] += count
                    else:
                        break
            dp.append(dp0)
        return sum(dp[-1].values())

    def outerTrees(self, trees: List[List[int]]) -> List[List[int]]:
        if len(trees) <= 1:
            return trees  # only 1 tree

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
        if k == 1:  # order will not be changed with any operation
            ans = s
            for i in range(len(s)):
                temp = s[i:] + s[:i]
                if temp < ans:
                    ans = temp
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
            if defend < max_defend:
                ans += 1
            else:
                max_defend = defend
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
            if root[x] != x:
                root[x] = find(root[x])
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
                if spf[i] != i:
                    continue  # Skip if it's a not prime number
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
        # print(root)
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
                    max_length[row][column][:2] = [
                        column - left, right - column]
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
        visited = set()  # element is nodes, for original nodes, we store '1'
        curr = []  # element is (step, node)
        heapq.heappush(curr, (0, 0))
        while curr:
            curr_node = heapq.heappop(curr)  # (step, node)
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
                            heapq.heappush(
                                curr, (curr_node[0] + count + 1, nei))
                            ans += count  # all subdivided nodes should be added to the answer
                            # we remove all sub-nodes from inter
                            inter[(curr_node[1], nei)] = 0
                            inter[(nei, curr_node[1])] = 0
                        else:
                            # no, we cannot reach this neighbor
                            # how many steps remain
                            temp = maxMoves - curr_node[0]
                            ans += temp
                            # we remove those sub-nodes from inter
                            inter[(curr_node[1], nei)] -= temp
                            inter[(nei, curr_node[1])] -= temp
                    else:
                        # this neighbor is visited before, just add subdivided nodes
                        ans += min(inter[(curr_node[1], nei)],
                                   maxMoves - curr_node[0])
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

    def maxNumberOfBalloons(self, text: str) -> int:
        # b a ll oo n
        ans = {'b': 0, 'a': 0, 'l': 0, 'o': 0, 'n': 0}
        for c in text:
            if c in ans:
                ans[c] += 1 if c in ['l', 'o'] else 2
        return min([i // 2 for i in ans.values()])

    def reverseOnlyLetters(self, s: str) -> str:
        c_list = list(s)
        left, right = 0, len(s) - 1
        while left < right:
            if not c_list[left].isalpha():
                left += 1
            elif not c_list[right].isalpha():
                right -= 1
            else:
                c_list[left], c_list[right] = c_list[right], c_list[left]
                left += 1
                right -= 1
        return "".join(c_list)

    def maxTurbulenceSize(self, arr: List[int]) -> int:
        # loop from left to right
        ans = count = pre = 0
        for i in arr:
            if count == 0:
                count += 1
            elif count == 1:
                if i != pre:
                    count += 1
                    bigger = (i < pre)
            # count >= 2
            elif (bigger and i > pre) or (not bigger and i < pre):
                count += 1
                bigger = not bigger
            elif i != pre:
                count = 2
                bigger = (i < pre)
            else:
                count = 1
            pre = i
            ans = max(ans, count)
        return ans

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        global m, n
        m, n = len(matrix), len(matrix[0])
        ans = []
        # right -> down -> left -> up -> ...

        def split_layer(x, y, step, dir):
            global m, n
            if step == 0:
                return
            for _ in range(step):
                x += dir[0]
                y += dir[1]
                ans.append(matrix[x][y])
            if dir == [0, 1]:  # right
                m -= 1
                split_layer(x, y, m, [1, 0])
            elif dir == [0, -1]:  # left
                m -= 1
                split_layer(x, y, m, [-1, 0])
            elif dir == [1, 0]:  # down
                n -= 1
                split_layer(x, y, n, [0, -1])
            elif dir == [-1, 0]:  # up
                n -= 1
                split_layer(x, y, n, [0, 1])
        split_layer(0, -1, n, [0, 1])
        return ans

    def addOperators(self, num: str, target: int) -> List[str]:
        # since num's len is less than 11
        # we can try brute force first
        ans = list()
        # def add_next(s: str, index: int):
        #     if index == len(s):
        #         # reach end
        #         try:
        #             if eval(s) == target:
        #                 ans.append(s)
        #         except: pass
        #     else:
        #         zero = False
        #         if s[index - 1] == '0':
        #             if index == 1: zero = True
        #             elif not s[index - 2].isdigit(): zero = True
        #         # none
        #         if not zero: add_next(s, index + 1)
        #         # '+'
        #         add_next(s[:index] + '+' + s[index:], index + 2)
        #         # '-'
        #         add_next(s[:index] + '-' + s[index:], index + 2)
        #         # '*'
        #         add_next(s[:index] + '*' + s[index:], index + 2)
        # add_next(num, 1)
        # return ans

        def backtracking(idx=0, path='', value=0, prev=None):
            if idx == len(num) and value == target:
                ans.append(path)
                return

            for i in range(idx + 1, len(num) + 1):
                tmp = int(num[idx: i])
                if i == idx + 1 or (i > idx + 1 and num[idx] != '0'):
                    if prev is None:
                        # no operator will be added
                        backtracking(i, num[idx: i], tmp, tmp)
                    else:
                        backtracking(i, path + '+' +
                                     num[idx: i], value + tmp, tmp)
                        backtracking(i, path + '-' +
                                     num[idx: i], value - tmp, -tmp)
                        backtracking(
                            i, path + '*' + num[idx: i], value - prev + prev * tmp, prev * tmp)
        ans = []
        backtracking()
        return ans

    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        ans = 0
        count = 0
        for i in nums:
            if i == 1:
                count += 1
                ans = max(ans, count)
            else:
                count = 0
        return ans

    def tribonacci(self, n: int) -> int:
        ans = [0, 1, 1]
        if n < len(ans):
            return ans[n]
        for _ in range(2, n):
            ans = [ans[1], ans[2], sum(ans)]
        return ans[2]

    def splitListToParts(self, head: ListNode, k: int) -> List[ListNode]:
        n, curr = 0, head
        while curr:
            curr = curr.next
            n += 1
        base = n // k
        addi = n - k * base

        # addi * (base + 1) + (k - addi) * base
        ans = list()
        root = curr = head
        for _ in range(addi):
            i = 1
            while i < base + 1:
                i += 1
                curr = curr.next
            ans.append(root)
            if curr:
                root = curr.next
                curr.next = None
                curr = root
            else:
                root = curr
        for _ in range(k - addi):
            i = 1
            while i < base:
                i += 1
                curr = curr.next
            ans.append(root)
            if curr:
                root = curr.next
                curr.next = None
                curr = root
        return ans

    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        if sum(nums) % k != 0:
            return False

        # can be equally subsets at first judgement
        def check_next(index, ans):
            # print(index, ans)
            if index == len(nums):
                return True
            curr = nums[index]
            for i in range(len(ans)):
                if ans[i] >= curr:
                    ans[i] -= curr
                    if check_next(index + 1, ans):
                        return True
                    ans[i] += curr
            return False

        ans = [sum(nums) // k] * k
        nums.sort(reverse=True)
        return check_next(0, ans)

    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        # dp
        m, n = len(dungeon), len(dungeon[0])
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if i == m - 1 and j == n - 1:
                    temp = 1 - dungeon[i][j]
                elif i == m - 1:
                    temp = dungeon[i][j + 1] - dungeon[i][j]
                elif j == n - 1:
                    temp = dungeon[i + 1][j] - dungeon[i][j]
                else:
                    temp = min(dungeon[i + 1][j], dungeon[i]
                               [j + 1]) - dungeon[i][j]
                dungeon[i][j] = max(temp, 1)
        return dungeon[0][0]

    def rob(self, nums: List[int]) -> int:
        # why do we need to record up to 3 steps previously?
        # consider this condition: 2, 1, 1, 2
        # the max money we can rob is 2 + 2
        pre = [0] * 3
        for i in nums:
            pre = [pre[1], pre[2], max(pre[0], pre[1]) + i]
        return max(pre)

    def rob_2(self, nums: List[int]) -> int:
        # houses are arranged in a circle
        # which means the first house is adjacent to the last one
        n = len(nums)
        if n == 1:
            return nums[0]
        # dp[0]: rob first, rob ith
        # dp[1]: rob first, not rob ith
        # dp[2]: not rob first, rob ith
        # dp[3]: not rob first, not rob ith
        dp = [0, nums[0], nums[1], 0]
        for i in range(2, n):
            dp = [dp[1] + nums[i], max(dp[0:2]), dp[3] + nums[i], max(dp[2:4])]
        return max(dp[1:4])

    def deleteAndEarn(self, nums: List[int]) -> int:
        dp = [0, 0]  # max if curr is picked, max if curr is not picked
        pre = -1
        count = collections.Counter(nums)
        for k in sorted(count.keys()):
            if k == pre + 1:  # current value is adjacent to the previous one
                dp = [dp[1] + k * count[k], max(dp)]
            else:
                # current value is not adjacent to the previous one
                dp = [max(dp) + k * count[k], max(dp)]
            pre = k
        return max(dp)

    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        # TLE

        # m, n = len(board), len(board[0])
        # vword = set()
        # vchar = set()
        # dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        # def find_next(x, y, word, index):
        #     vchar.add((x, y))
        #     if index == len(word): return True
        #     target = word[index]
        #     for i, j in dir:
        #         x_, y_ = x + i, y + j
        #         if m > x_ >= 0 <= y_ < n and (x_, y_) not in vchar:
        #             vword.add(word[:index] + board[x_][y_])
        #             if board[x_][y_] == target:
        #                 if find_next(x_, y_, word, index + 1): return True
        #     vchar.remove((x, y))
        #     return False

        # ans = []
        # for i in words:
        #     if i in vword:
        #         ans.append(i)
        #         continue
        #     vchar.clear()
        #     if any(board[x][y] == i[0] and find_next(x, y, i, 1) for x in range(m) for y in range(n)):
        #         ans.append(i)
        # return ans

        # time complexity is too high

        # m, n = len(board), len(board[0])
        # vword = set()
        # vchar = set()
        # dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        # # DFS search
        # # since 1 <= len(words[i]) <= 10
        # def search(x, y, word):
        #     vchar.add((x, y))
        #     word += board[x][y]
        #     vword.add(word)
        #     if len(word) < 10:
        #         for i, j in dir:
        #             x_, y_ = x + i, y + j
        #             if m > x_ >= 0 <= y_ < n and (x_, y_) not in vchar:
        #                 search(x_, y_, word)
        #     word = word[:-1]
        #     vchar.remove((x, y))
        # for i in range(m):
        #     for j in range(n):
        #         vchar.clear()
        #         search(i, j, "")
        # return [i for i in words if i in vword]

        # try trie structure
        t = Trie()
        for i in words:
            t.insert(i)

        m, n = len(board), len(board[0])
        ans = list()
        dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        def search(bcopy, x, y, tree):
            c = bcopy[x][y]
            # cannot find words
            if c == '-' or c not in tree:
                return

            # search forward
            tree = tree[c]
            if '%' in tree:
                ans.append(tree['%'])
                tree.pop('%')  # avoid duplicate result for a word
            bcopy[x][y] = '-'
            for i, j in dir:
                x_, y_ = x + i, y + j
                if m > x_ >= 0 <= y_ < n:
                    search(bcopy, x_, y_, tree)
            bcopy[x][y] = c

        for i in range(m):
            for j in range(n):
                search(board.copy(), i, j, t.tree)
        return ans

    def canJump(self, nums: List[int]) -> bool:
        target = len(nums) - 1
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] + i >= target:
                target = i
            if target == 0:
                return True
        return False

    def jump(self, nums: List[int]) -> int:
        # DFS
        # visited = set()
        # visited.add(0)
        # def jump(step, curr):
        #     next_pos = set()
        #     for pos in curr:
        #         if pos == len(nums) - 1: return step
        #         for next in range(pos + 1, pos + nums[pos] + 1):
        #             if next not in visited and next < len(nums):
        #                 visited.add(next)
        #                 next_pos.add(next)
        #     return jump(step + 1, next_pos)
        # return jump(0, {0})

        # heapq
        # h = []
        # heapq.heappush(h, (0, 0)) # element: [step, -index]
        # visited = set()
        # while True:
        #     temp = heapq.heappop(h)
        #     # reverse -index to normal index
        #     step = temp[0]
        #     index = -temp[1]
        #     if index == len(nums) - 1: return step
        #     if index in visited: continue
        #     visited.add(index)
        #     for next in range(index + 1, min(index + nums[index] + 1, len(nums))):
        #         heapq.heappush(h, (step + 1, -next))

        left = right = step = 0
        while True:
            if right >= len(nums) - 1:
                return step
            left, right = right + \
                1, max(i + nums[i] for i in range(left, right + 1))
            step += 1

    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        # how to calculate distance between 2 leaves, l1 & l2:
        # find their lowest common-parent node in i level
        # l1 at i1 level, while l2 at i2 level
        # their distance is i1 - i + i2 - i
        # Hence, we need to sum up depths of left and right subtrees of each node
        # And return the biggest sum
        global ans
        ans = 0

        def depth(node: TreeNode):
            global ans
            if not node:
                return -1
            l, r = depth(node.left), depth(node.right)
            ans = max(ans, l + r + 2)
            return max(l, r) + 1
        depth(root)
        return ans

    def maxProduct(self, nums: List[int]) -> int:
        # when we find a zero, we restart calculation
        max_posi, max_nega = 0, 0
        ans = nums[0]
        for i in nums:
            if i == 0:
                max_posi, max_nega = 0, 0
            if i > 0:
                max_posi, max_nega = i * max_posi if max_posi != 0 else i, i * \
                    max_nega if max_nega != 0 else 0
            else:
                max_posi, max_nega = i * max_nega if max_nega != 0 else 0, i * \
                    max_posi if max_posi != 0 else i
            ans = max(ans, max_posi if max_posi > 0 else max_nega)
        return ans

    def getMaxLen(self, nums: List[int]) -> int:
        nums = [0] + nums + [0]
        ans = nega_count = zero_index = 0
        for i in range(len(nums)):
            if nums[i] < 0:
                if first_nega == -1:
                    first_nega = i
                last_nega, nega_count = i, nega_count + 1
            if nums[i] == 0:
                if nega_count % 2 == 0:
                    # even number of negative integer
                    ans = max(ans, i - 1 - zero_index)
                else:
                    # odd number of negative integer
                    ans = max(ans, max(i - 1 - first_nega,
                              last_nega - zero_index - 1))
                zero_index, nega_count = i, 0
                first_nega = last_nega = -1
        return ans

    # The guess API is already defined for you.
    # @param num, your guess
    # @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
    # def guess(num: int) -> int:
    def guessNumber(self, n: int) -> int:
        def guess(i):
            return 1
        left, right = 1, n
        while right > left:
            mid = left + (right - left) // 2
            temp = guess(mid)
            if temp == -1:
                right = mid - 1
            elif temp == 1:
                left = mid + 1
            else:
                return mid
        return left

    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        max_left = values[0]
        ans = -float('inf')
        for i in range(1, len(values)):
            ans = max(ans, values[i] - i + max_left)
            max_left = max(max_left, values[i] + i)
        return ans

    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        lowest = prices[0]
        for i in prices:
            ans = max(ans, i - lowest)
            lowest = min(lowest, i)
        return ans

    def maxProfitCooldown(self, prices: List[int]) -> int:
        """
        dp[i] = [dp[i][0]->rest[i], dp[i][1]->hold[i], dp[i][2]->sell[i]]
        rest[i]: no action = max(rest[i - 1], sell[i - 1])
        hold[i]: own one stock = max(rest[i - 1] - prices[i], hold[i - 1])
        sell[i]: sell stock = hold[i - 1] + prices[i]
        """
        dp = [0, -prices[0], -float('inf')]
        for p in prices[1:]:
            dp = [max(dp[0], dp[2]), max(dp[0] - p, dp[1]), dp[1] + p]
        return max(dp)

    def maxProfitFee(self, prices: List[int], fee: int) -> int:
        """
        dp[i] = [dp[i][0]->free[i], dp[i][1]->hold[i]]
        free[i]: free hand = max(free[i - 1], hold[i - 1] + prices[i] - fee)
        hold[i]: own stock = max(free[i - 1] - prices[i], hold[i - 1])
        """
        dp = [0, - prices[0]]
        for p in prices[1:]:
            dp = [max(dp[0], dp[1] + p - fee), max(dp[0] - p, dp[1])]
        return max(dp)

    def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
        def subtree(start, end):
            if start >= end:
                return None
            root = TreeNode(preorder[start])
            mid = end
            for i in range(start + 1, end):
                if preorder[i] > root.val:
                    mid = i
                    break
            root.left = subtree(start + 1, mid)
            root.right = subtree(mid, end)
            return root
        return subtree(0, len(preorder))

    def numSquares(self, n: int) -> int:
        dp = [0]
        while len(dp) <= n:
            curr = len(dp)
            dp.append(min(dp[curr - i * i]
                      for i in range(1, int(curr ** 0.5 + 1))) + 1)
        return dp[-1]

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # TLE
        # def next_break(word):
        #     if len(word) == 0 or word in wordDict: return True
        #     return any(word[:i + 1] in wordDict and next_break(word[i + 1:]) for i in range(len(word)))
        # return next_break(s)

        # dp
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        for i in range(n):
            dp[i + 1] = True if any(s[mid: i + 1] in wordDict and dp[mid]
                                    for mid in range(i, -1, -1)) else False
        return dp[-1]

    def trap(self, height: List[int]) -> int:
        # TLE
        # n = len(height)
        # dp = [[0] * 2 for _ in range(n)]
        # for i in range(len(height)):
        #     curr = height[i]
        #     # update right max for i in range(i)
        #     # update left max for i in range(i, n)
        #     for idx in range(i): dp[idx][1] = max(dp[idx][1], curr)
        #     for idx in range(i, n): dp[idx][0] = max(dp[idx][0], curr)
        # return sum([max(min(dp[i]) - height[i], 0) for i in range(n)])

        curve = list()
        n = len(height)
        left_max = right_max = 0
        for i in range(n):
            if height[i] > left_max:
                curve.append(i)
                left_max = height[i]
            if height[-1-i] > right_max:
                curve.append(n - i - 1)
                right_max = height[-1-i]
        curve = sorted(list(set(curve)))
        return sum((min(height[left], height[right]) - height[i]) for left, right in zip(curve, curve[1:]) for i in range(left + 1, right))

    def maxProfitIII(self, prices: List[int]) -> int:
        # dp
        """
        state machine:
        init -> own -> empty -> own -> done
        """
        ans = 0
        dp = [0, -float('inf'), -float('inf'), -float('inf'), -float('inf')]
        for p in prices:
            dp = [0, max(-p, dp[1]), max(dp[1] + p, dp[2]),
                  max(dp[2] - p, dp[3]), max(dp[3] + p, dp[4])]
            ans = max(max(dp), ans)
            # print(dp)
        return ans

    def numTrees(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1  # 0个数字，只有一种情况; 1个数字，也只有一种情况
        # for a contiguous number list from start to end, let's say end - start is i
        # we calculate number of BST for this case, and record it in dp[i]
        for i in range(1, n):
            for j in range(i):
                dp[i + 1] += dp[j] * dp[i - j]
        return dp[-1]

    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        ans = list()

        def find_child(node, depth, parent):
            if not node:
                return
            if node.val in [x, y]:
                ans.append([depth, parent])
            find_child(node.left, depth + 1, node.val)
            find_child(node.right, depth + 1, node.val)
        find_child(root, 0, 0)
        if len(ans) != 2:
            return False
        x, y = ans[0], ans[1]
        return True if x[0] == y[0] and x[1] != y[1] else False

    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        # dp
        # first row, no need to modify
        m, n = len(matrix), len(matrix[0])
        for i in range(1, m):
            for j in range(n):
                matrix[i][j] += min([matrix[i - 1][k]
                                    for k in range(max(0, j - 1), min(n, j + 2))])
        return min(matrix[-1])

    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # dp
        # first row, remain original
        n = len(triangle)
        for i in range(1, n):
            for j in range(i + 1):
                """
                for ith row, we have i + 1 elements. indexed from 0 to i
                thus, for previous row, we have i elements, indexed from 0 to i - 1
                """
                triangle[i][j] += min([triangle[i - 1][k]
                                      for k in range(max(0, j - 1), min(i, j + 1))])
        return min(triangle[-1])

    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        pending = dict()
        ans = [-1] * len(nums1)
        # (idx in nums2, idx in nums1)
        n1 = sorted([(nums2.index(nums1[i]), i) for i in range(len(nums1))])
        for i in range(len(nums2)):
            if n1 and i == n1[0][0]:
                temp = n1.pop(0)
                pending[nums2[temp[0]]] = temp[1]
            for value, idx in pending.items():
                if nums2[i] > value and ans[idx] == -1:
                    ans[idx] = nums2[i]
        return ans

    def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
        # dp[x][y] represents for total sum of cells range from [0][0] to [x][y]
        # for a rectangle area: top-left cell - (x1, y1) | bottom-right cell - (x2, y2)
        # total sum = dp[x2, y2] - dp[x1, y2] - dp[x2, y1] + dp[x1, y1]
        m, n = len(mat), len(mat[0])
        for i in range(m):
            for j in range(n):
                mat[i][j] += (mat[i - 1][j] if i > 0 else 0) + (mat[i][j - 1]
                                                                if j > 0 else 0) - (mat[i - 1][j - 1] if i > 0 and j > 0 else 0)
        ans = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                x1, y1, x2, y2 = max(-1, i - k - 1), max(-1, j -
                                                         k - 1), min(m - 1, i + k), min(n - 1, j + k)
                # print(i, j)
                # print(x1, y1, x2, y2)
                ans[i][j] = mat[x2][y2] - (mat[x1][y2] if x1 >= 0 else 0) - (
                    mat[x2][y1] if y1 >= 0 else 0) + (mat[x1][y1] if x1 >= 0 and y1 >= 0 else 0)
        return ans

    def reverseWords(self, s: str) -> str:
        ans = list(filter(None, s.split(' ')))
        ans.reverse()
        return ' '.join(ans)

    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1] * n
        for i in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        # obstacles
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [0] * n
        for i in range(m):
            dp[0] = 1 - obstacleGrid[i][0] if i == 0 else 1 if dp[0] == 1 and obstacleGrid[i][0] == 0 else 0
            for j in range(1, n):
                dp[j] = dp[j] + dp[j - 1] if obstacleGrid[i][j] == 0 else 0
            # print(dp)
        return dp[-1]

    def frequencySort(self, s: str) -> str:
        occur = collections.Counter(s)
        ans = ""
        for ch, count in reversed(sorted(occur.items(), key=lambda x: x[1])):
            ans += ch * count
        return ans

    def longestPalindrome(self, s: str) -> str:
        # dp
        n = len(s)
        ans = s[0]
        dp = [[-float('inf')] * (n + 1) for _ in range(n)]
        for i in range(n):
            dp[i][i] = 0
            dp[i][i + 1] = 1
        for i in range(2, n + 1):
            for left in range(n - i + 1):
                right = left + i
                if s[left] == s[right - 1]:
                    dp[left][right] = dp[left + 1][right - 1] + 2
                    if dp[left][right] > len(ans):
                        ans = s[left:right]
        return ans

    def longestPalindromeSubseq(self, s: str) -> int:
        # dp
        n = len(s)
        dp = [[0] * (n + 1) for _ in range(n)]
        for i in range(n):
            dp[i][i + 1] = 1
        for i in range(2, n + 1):
            for left in range(n - i + 1):
                right = left + i
                if s[left] == s[right - 1]:
                    dp[left][right] = dp[left + 1][right - 1] + 2
                else:
                    dp[left][right] = max(
                        dp[left + 1][right], dp[left][right - 1])
        # print(dp)
        return max([max(i) for i in dp])

    def coinChange(self, coins: List[int], amount: int) -> int:
        # dp
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
            dp[i] = min([dp[i]] + [1 + dp[i - coin]
                        for coin in coins if i - coin >= 0])
        return dp[-1] if dp[-1] != float('inf') else -1

    def change(self, amount: int, coins: List[int]) -> int:
        # 需要计算所有不同组合，值得注意的是：如果我们先遍历amount，在遍历coins，会导致同一个组合被重复计算
        # 例如：参数为6, [2, 4]，dp[6] = dp[4] + dp[2] = 2 + 1 = 3
        # 4 = 4 = 2 + 2 -> 均进行+2操作 -> 6 = 4 + 2 = 2 + 2 + 2
        # 2 = 2 -> 均进行+4操作 -> 6 = 2 + 4
        # 这里重复计算了2 + 4的组合
        #
        # 因此，我们考虑另一种遍历方式，首先遍历coins，再遍历amount。这样同一面值的coin的个数永远不会被重复计算
        # 例如：参数为6, [2, 4]，初始化dp = [1, 0, 0, 0, 0, 0, 0]
        # 先遍历硬币2 dp = [1, 0, 1, 0, 1, 0, 1]
        # 在遍历硬币4 dp = [1, 0, 1, 0, 2, 0, 2] 最终结果为2，正确
        # dp
        dp = [1] + [0] * amount
        for c in coins:
            for i in range(c, amount + 1):
                dp[i] += dp[i - c]
        return dp[-1]

    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        count = collections.Counter(nums)
        idx = 0
        for value, count[value] in sorted(count.keys()):
            for i in count[value]:
                nums[idx] = value
                idx += 1
        return nums

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # traverse first number, then we can get target sum of the other two numbers
        # use left, right to find all combinations
        n = len(nums)
        nums.sort()
        ans = list()
        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            target = 0 - nums[i]
            left, right = i + 1, n - 1
            while left < right:
                if left > i + 1 and nums[left] == nums[left - 1]:
                    left += 1
                elif right < n - 1 and nums[right] == nums[right + 1]:
                    right -= 1
                elif nums[left] + nums[right] == target:
                    ans.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1
                elif nums[left] + nums[right] > target:
                    right -= 1
                else:
                    left += 1
        return ans

    def orangesRotting(self, grid: List[List[int]]) -> int:
        # frash orange will become rotten if it is adjacent to a rotten orange
        # second approach
        dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        m, n = len(grid), len(grid[0])
        ans = 0
        rotten = collections.deque([[i, j] for i, j in product(range(m), range(n)) if grid[i][j] == 2])
        cnt = sum([1 for i, j in product(range(m), range(n)) if grid[i][j] == 1])

        while rotten:
            size = len(rotten)
            for _ in range(size):
                x, y = rotten.popleft()
                for i, j in dir:
                    xi, yj = x + i, y + j
                    if m > xi >= 0 <= yj < n and grid[xi][yj] == 1:
                        grid[xi][yj] = 2
                        cnt -= 1
                        rotten.append([xi, yj])
            if rotten: ans += 1

        return ans if cnt == 0 else -1

        # beats over 97% in time
        rotten = list()
        frash = set()
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    rotten.append((i, j))
                if grid[i][j] == 1:
                    frash.add((i, j))

        dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        ans = 0
        while frash:
            if not rotten:
                return -1
            amount = len(rotten)
            for i in range(amount):
                curr = rotten[i]
                for x, y in dir:
                    x_, y_ = x + curr[0], y + curr[1]
                    if (x_, y_) in frash:
                        frash.remove((x_, y_))
                        rotten.append((x_, y_))
            rotten = rotten[amount:]
            ans += 1
        return ans

    def countValidWords(self, sentence: str) -> int:
        words = sentence.split(' ')
        ans = 0
        for w in words:
            if w:
                n = len(w)
                dash_count = 0
                for i in range(n):
                    if w[i] == '-':
                        if 0 < i < n - 1 and w[i - 1].isalpha() and w[i + 1].isalpha():
                            dash_count += 1
                        else:
                            ans -= 1
                            break
                    if w[i].isdigit() or (w[i] in [',', '!', '.'] and i != n - 1) or dash_count > 1:
                        ans -= 1
                        break
                ans += 1
        return ans

    def nextBeautifulNumber(self, n: int) -> int:
        # digit i has an occurance i
        all_number = set()
        all_poss = ["1", "22", "122", "333", "1333", "4444", "14444",
                    "22333", "55555", "155555", "122333", "224444", "666666"]

        def find_all(num_dict, pre):
            if max(num_dict.values()) == 0:
                [all_number.add(p) for p in pre]
            for k, v in num_dict.items():
                if v > 0:
                    num_dict[k] -= 1
                    temp = set()
                    for p in pre:
                        temp.add(p + k)
                    find_all(num_dict, temp)
                    num_dict[k] += 1
        for poss in all_poss:
            find_all(collections.Counter(poss), [""])
        all_number.add("1224444")
        all_number = sorted([int(i) for i in all_number])

        left, right = 0, len(all_number) - 1
        while left < right:
            mid = left + (right - left) // 2
            if all_number[mid] <= n:
                left = mid + 1
            if all_number[mid] > n:
                right = mid
        return all_number[left]

    def countHighestScoreNodes(self, parents: List[int]) -> int:
        # remove one node
        # first, we need to know total number of nodes
        # then, we need to calculate subtrees of this node
        n = len(parents)
        global ans, max_prod
        ans, max_prod = 0, 0

        # convert parents to node - children dict
        tree = collections.defaultdict(list)
        for i in range(n):
            tree[parents[i]].append(i)

        def find_subtree(node):
            global ans, max_prod
            children = tree[node]
            amount = 1
            prod = 1
            for ch in children:
                count = find_subtree(ch)
                amount += count
                prod *= max(count, 1)
            prod *= max(n - amount, 1)
            if prod > max_prod:
                max_prod = prod
                ans = 1
            elif prod == max_prod:
                ans += 1
            return amount
        find_subtree(0)
        return ans

    def minimumTime(self, n: int, relations: List[List[int]], time: List[int]) -> int:
        # form a nextcourse, precourse dict
        course_dep = collections.defaultdict(list)
        final_course = set([i + 1 for i in range(n)])
        for pre, next in relations:
            course_dep[next].append(pre)
            if pre in final_course:
                final_course.remove(pre)
        global ans
        ans = 0

        visited = collections.defaultdict(int)

        def count_next(course, month):
            # prune it if this course has been visited with a bigger cost
            if month <= visited[course]:
                return
            visited[course] = month
            global ans
            if not course_dep[course]:
                ans = max(ans, month)
                return
            for c in course_dep[course]:
                count_next(c, month + time[c - 1])
        for c in final_course:
            # DFS
            count_next(c, time[c - 1])
        return ans

    def solveChess(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        # chess game
        # for boarder cells, they are not counted as lost points
        m, n = len(board), len(board[0])
        remain = set()
        next = list()
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    remain.add((i, j))
                    if i in [0, m - 1] or j in [0, n - 1]:
                        next.append((i, j))
        dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        while next:
            next = list(set(next))
            ne = len(next)
            for _ in range(ne):
                (x, y) = next.pop()
                remain.discard((x, y))
                for i, j in dir:
                    x_, y_ = x + i, y + j
                    if m > x_ >= 0 <= y_ < n and board[x_][y_] == 'O' and (x_, y_) in remain:
                        next.append((x_, y_))
        for i, j in remain:
            board[i][j] = 'X'
        return board
        # leetcode trick: it will get result from original address
        # if we re-point board to board_, the original board does not change value

    def nodesBetweenCriticalPoints(self, head: ListNode) -> List[int]:
        minD, maxD = float('inf'), -1
        critical = list()
        if not head:
            return [-1, -1]
        pre = head.val
        curr = head.next
        idx = 1
        while curr and curr.next:
            if pre < curr.val > curr.next.val or pre > curr.val < curr.next.val:
                critical.append(idx)
                if len(critical) > 1:
                    minD = min(minD, idx - critical[-2])
                    maxD = idx - critical[0]
            pre = curr.val
            curr = curr.next
            idx += 1
        if len(critical) < 2:
            return [-1, -1]
        return[minD, maxD]

    def minimumOperations(self, nums: List[int], start: int, goal: int) -> int:
        visited = set()
        next = list()
        next.append((start, 0))
        while next:
            (curr, step) = next.pop(0)
            step += 1
            for i in nums:
                temp = [curr + i, curr - i, curr ^ i]
                for t in temp:
                    if t == goal:
                        return step
                    if 0 <= t <= 1000 and t not in visited:
                        visited.add(t)
                        next.append((t, step))
        return -1

    def possiblyEquals(self, s1: str, s2: str) -> bool:
        # correct but TLE :(
        # all_comb = set()
        # global ans
        # ans = False
        # visited = set()
        # def compare_decode(ss1, ss2):
        #     sp1, sp2 = ss1.split(',')[1:], ss2.split(',')[1:]
        #     letter1, letter2 = dict(), dict()
        #     idx1 = idx2 = 0
        #     for c in sp1:
        #         if c.isalpha():
        #             letter1[idx1] = c
        #         else:
        #             idx1 += int(c)
        #     for c in sp2:
        #         if c.isalpha():
        #             letter2[idx2] = c
        #         else:
        #             idx2 += int(c)
        #     if idx1 != idx2: return False
        #     for k, v in letter1.items():
        #         if k in letter2 and v != letter2[k]: return False
        #     return True

        # def find_next(s: str, idx: int, pre_num: int, pre_str: str):
        #     global ans
        #     if ans: return
        #     if idx == len(s):
        #         res = pre_str + (f',{pre_num}' if pre_num > 0 else '')
        #         if second:
        #             if res in visited: return
        #             ans = any(compare_decode(res, i) for i in all_comb)
        #             visited.add(res)
        #         else: all_comb.add(res)
        #         return
        #     curr = s[idx]
        #     if curr.isalpha(): find_next(s, idx + 1, 0, pre_str + (f',{pre_num},' if pre_num > 0 else ',') + curr) # letter
        #     else: # number
        #         # two decode ways
        #         if pre_num != 0: # we can only split string into non-empty substrings
        #             find_next(s, idx + 1, pre_num * 10 + int(curr), pre_str) # concatenate current digit to previous number
        #         if int(curr) != 0:
        #             find_next(s, idx + 1, int(curr), pre_str + (f',{pre_num}' if pre_num > 0 else '')) # start a new number

        # second = False
        # find_next(s1, 0, 0, "")
        # print(all_comb)
        # second = True
        # find_next(s2, 0, 0, "")
        # return ans

        # from discussion
        # DP
        def calc_length(s):
            """Return possible length."""
            # given a number, return all possible lengths it can be decoded
            # for example, 25 can be decoded to 7 and 25
            ans = {int(s)}
            for i in range(1, len(s)):
                # merge two lists without duplicate items
                ans |= {x + y for x in calc_length(s[:i])
                        for y in calc_length(s[i:])}
            return ans

        @functools.cache
        # idx of s1, idx of s2, diff is the amount len(s2.decode) - len(s1.decode)
        def find_next(i, j, diff):
            """Return True if s1[i:] matches s2[j:] with given differences."""
            if i == len(s1) and j == len(s2):
                return diff == 0  # reached end, with no different left
            if i < len(s1) and s1[i].isdigit():
                # s1[i] is a digit, we need to find the whole number and record all possible lengths it can be decoded
                ii = i
                while ii < len(s1) and s1[ii].isdigit():
                    ii += 1
                # s1[i:ii] is the whole number
                # gg(s1[i:ii]) returns a list consists of all possible lengths
                for x in calc_length(s1[i:ii]):
                    if find_next(ii, j, diff - x):
                        return True
            elif j < len(s2) and s2[j].isdigit():
                # same thing to s2
                jj = j
                while jj < len(s2) and s2[jj].isdigit():
                    jj += 1
                for x in calc_length(s2[j:jj]):
                    if find_next(i, jj, diff + x):
                        return True
            elif diff == 0:
                # first, they have the same length
                # then, s1[i] and s2[j] are letters
                # further more, they are the same letter
                # then we can move both of them one index forward
                if i < len(s1) and j < len(s2) and s1[i] == s2[j]:
                    return find_next(i + 1, j + 1, 0)
            elif diff > 0:
                # diff > 0: len(s2.decode) - len(s1.decode), s1 should be added more length to chase s2
                # since current item is alpha, it has length of 1
                # we simply decrease diff by 1, and move one step forward of s1
                if i < len(s1):
                    return find_next(i + 1, j, diff - 1)
            else:
                # diff < 0
                if j < len(s2):
                    return find_next(i, j + 1, diff + 1)
            return False  # return False if all conditions are not accepted

        return find_next(0, 0, 0)

    def sumNumbers(self, root: TreeNode) -> int:
        global ans
        ans = 0

        def find_leaf(node: TreeNode, pre: int):
            global ans
            if not node.left and not node.right:
                ans += pre * 10 + node.val
                return
            pre *= 10
            pre += node.val
            if node.left:
                find_leaf(node.left, pre)
            if node.right:
                find_leaf(node.right, pre)
            return
        find_leaf(root, 0)
        return ans

    # study plan: two pointers
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        nonzero_index = list()
        zero_count = 0
        for i in nums:
            if i == 0:
                zero_count += 1
            else:
                nonzero_index.append(i)

        nums[:] = nonzero_index + [0] * (n - len(nonzero_index))

    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        # exactly one solution is garanteed
        left, right = 0, len(numbers) - 1
        while left < right:
            curr = numbers[left] + numbers[right]
            if curr == target:
                return [left + 1, right + 1]
            if curr < target:
                left += 1
            else:
                right -= 1
        return [1, 1]

    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        # sum up all left leaves(not node!!! we count it only if it is a leaf and it is left child)
        def find_left(node: TreeNode, left: bool):
            if not node.left and not node.right:
                return 0 if not left else node.val
            return (find_left(node.left, True) if node.left else 0) + (find_left(node.right, False) if node.right else 0)

        return find_left(root, False)

    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left, right = 0, len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1

    def reverseWords(self, s: str) -> str:
        return ' '.join([i[::-1] for i in s.split(' ')])

    def kthDistinct(self, arr: List[str], k: int) -> str:
        count = collections.Counter(arr)
        for a in arr:
            if count[a] == 1:
                k -= 1
                if k == 0:
                    return a
        return ""

    def maxTwoEvents(self, events: List[List[int]]) -> int:
        # dp
        # events.sort(key=lambda x: x[1]) # sort with end time
        # dp = [[0] * 3] # [attend 0 event, 1 event, 2 events]
        # idx = 0
        # curr_event = events[idx]
        # for i in range(1, events[-1][1] + 1):
        #     dp.append(dp[-1])
        #     if i == curr_event[1]:
        #         # current event is ended
        #         # update dp
        #         while i == curr_event[1]:
        #             dp[-1] = [dp[-1][0], max(dp[-1][1], curr_event[2]), max(dp[-1][2], dp[curr_event[0] - 1][1] + curr_event[2])]
        #             idx += 1
        #             if idx == len(events): break
        #             curr_event = events[idx]
        # return max(dp[-1])
        # one more constraint: you can take at most two events

        # using heap
        events.sort(key=lambda x: x[0])  # sort with start time

        hq = []
        pop_off_max = 0
        ans = 0
        for e in events:
            while hq and hq[0][0] < e[0]:
                pop_off_max = max(pop_off_max, heapq.heappop(hq)[1])
            ans = max(pop_off_max + e[2], ans)
            heapq.heappush(hq, e[1:])
        return ans

    def platesBetweenCandles(self, s: str, queries: List[List[int]]) -> List[int]:
        n = len(s)
        candle = []
        for i in range(n):
            if s[i] == '|':
                candle.append(i)

        def find_index(k, r: bool):
            left, right = 0, len(candle) - 1
            while left <= right:
                mid = left + (right - left) // 2
                if candle[mid] > k:
                    right = mid - 1
                elif candle[mid] < k:
                    left = mid + 1
                else:
                    return mid if not r else mid + 1
            return left
        ans = []
        for x, y in queries:
            start = find_index(x, False)
            end = find_index(y, True) - 1
            if start > end:
                ans.append(0)
            else:
                ans.append(candle[end] - candle[start] - (end - start))
        return ans

    def arrangeCoins(self, n: int) -> int:
        ans = 0
        while n > ans:
            n -= ans + 1
            ans += 1
        return ans

    def middleNode(self, head: ListNode) -> ListNode:
        mid = curr = head
        while curr and curr.next:
            curr = curr.next.next
            mid = mid.next
        return mid

    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        curr = head
        while n > 0:
            curr = curr.next
            n -= 1
        if not curr:
            return head.next
        else:
            curr = curr.next
            pre = head
            remove = pre.next
        while curr:
            curr = curr.next
            pre = pre.next
            remove = remove.next
        # pre -> remove -> remove.next
        # we need to delete 'remove' node
        pre.next = remove.next
        return head

    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
        def merge_sub(node1, node2):
            if not node1:
                return node2
            if not node2:
                return node1
            node1.val += node2.val
            node1.left = merge_sub(node1.left, node2.left)
            node1.right = merge_sub(node1.right, node2.right)
            return node1

        return merge_sub(root1, root2)

    def connect(self, root: 'Node') -> 'Node':
        pre = []

        def search_sub(node, level):
            if not node:
                return
            if level == len(pre):
                pre.append(node)
            else:
                pre[level].next = node
                pre[level] = node
            search_sub(node.left, level + 1)
            search_sub(node.right, level + 1)

        search_sub(root, 0)
        return root

    def countVowelSubstrings(self, word: str) -> int:
        # consist only vowel chars and all present
        vowel = {'a': 0, 'e': 0, 'i': 0, 'o': 0, 'u': 0}
        curr = list()
        ans = 0
        for c in word:
            # print(curr)
            if c in vowel:
                for cu in curr:
                    cu[c] += 1
                    if min(cu.values()) > 0:
                        ans += 1
                vowel_ = vowel.copy()
                vowel_[c] += 1
                curr.append(vowel_)
            else:
                curr.clear()
        return ans

    def countVowels(self, word: str) -> int:
        n = len(word)
        ans = 0
        for i in range(n):
            if word[i] in ['a', 'e', 'i', 'o', 'u']:
                # left end: 0 ~ i; right end: i + 1 ~ n
                ans += (i + 1) * (n - i)
        return ans

    def minimizedMaximum(self, n: int, quantities: List[int]) -> int:
        # if n == 1: return quantities[0]
        # m = len(quantities)
        # if n == m: return max(quantities)
        # quantities.sort(reverse=True)
        # perfect = math.ceil(sum(quantities) / n)

        # temp_n = n
        # idx = 0
        # while idx < m:
        #     if temp_n < m - idx:
        #         perfect += 1
        #         idx = 0
        #         temp_n = n
        #     else:
        #         count = math.ceil(quantities[idx] / perfect)
        #         if count > temp_n:
        #             perfect += 1
        #             idx = 0
        #             temp_n = n
        #         else:
        #             temp_n -= count
        #             idx += 1
        # return perfect
        # TLE, the point is you are increasing perfect by 1 every time, and it will cost a huge amount of time when you are face a big input array
        # here, the first thought is using binary search to reach target in O(logN) time complexity
        left, right = 1, max(quantities)
        while left < right:
            mid = left + (right - left) // 2
            if sum([math.ceil(i / mid) for i in quantities]) <= n:
                right = mid
            else:
                left = mid + 1
        return left
        # 3, [2,10,6] | 4, [2,2,8,7] | 22, [25,11,29,6,24,4,29,18,6,13,25,30]

    def maximalPathQuality(self, values: List[int], edges: List[List[int]], maxTime: int) -> int:
        # huge problem
        # start from 0, and end at 0
        # 1 - brute force
        # DFS: end until we ran out of time
        # and record ans when we hit 0
        global ans
        ans = 0
        edge = collections.defaultdict(list)
        path = [0]
        for x, y, c in edges:
            edge[x].append([y, c])
            edge[y].append([x, c])

        def move(node, time_left):
            global ans
            if node == 0:
                ans = max(ans, sum([values[i] for i in set(path)]))
            for i, j in edge[node]:
                if j <= time_left:
                    path.append(i)
                    move(i, time_left - j)
                    path.pop()
        move(0, maxTime)
        return ans

    def maxProfitII(self, prices: List[int]) -> int:
        ans = 0
        prices = [float('inf')] + prices + [-float('inf')]
        for i in range(1, len(prices) - 1):
            if prices[i - 1] >= prices[i] < prices[i + 1]:
                ans -= prices[i]
            if prices[i - 1] < prices[i] >= prices[i + 1]:
                ans += prices[i]
        return ans

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        def merge(node1, node2):
            if not node1:
                return node2
            if not node2:
                return node1
            if node1.val <= node2.val:
                node1.next = merge(node1.next, node2)
                return node1
            else:
                node2.next = merge(node1, node2.next)
                return node2

        return merge(l1, l2)

    def reverseList(self, head: ListNode) -> ListNode:
        if not head:
            return head
        pre, curr, ne = None, head, head.next
        while curr:
            curr.next = pre
            pre = curr
            curr = ne
            if ne:
                ne = ne.next
        return pre

    def countCombinations(self, pieces: List[str], positions: List[List[int]]) -> int:
        positions = [tuple(x) for x in positions]
        ans = set()

        def dfs(pos, dirs, stopped_mask):
            if stopped_mask == 0:
                return  # all roles are stoppped
            ans.add(tuple(pos))  # add current position
            # len(dirs) = len(pieces) 有几个棋子就有几个行走方向
            for active in range(1 << len(dirs)):
                """
                首先，对于第一次运行到此处的代码，所有棋子的状态应该都是仍在行走中，即stopped_mask全为1
                接下去，我们需要做的是，遍历棋子是否继续保持行走状态的选项。
                例如，有2个棋子，在不考虑棋子当前状态的情况下，对于接下去的状态，共有2**2 = 1<<2 = 4种可能性。
                即：棋子1行走，2行走；棋子1行走，2停止；棋子1停止，2行走；棋子1停止，2停止。
                如何判断某一种选项是否可行？
                stopped_mask & active != active
                如果该等式成立，则表示对于某一个棋子，在stopped_mask中为0状态（停止），而在active中为1状态（行走）。
                由于一个棋子一旦停止了，无法再次开始行走，所以该选项与当前状态不兼容。
                即在当前stopped_mask状态下，无法变更到active状态，所以需要跳过该选项。
                """
                if stopped_mask & active != active:
                    continue
                new_pos = list(pos)
                """
                原答案中此处使用了异或操作，经过分析，原答案中对active的定义为，下一个状态是否发生变化。
                即上面排除的情况为，当前状态为0（停止），下一状态发生变化，由于棋子停止后状态无法发生变化，故排除。
                当前状态1，状态变化1，异或 = 0 
                当前状态1，状态不变0，异或 = 1
                当前状态0，状态不变0，异或 = 0

                我们也可以将active直接定义为下一状态的值。
                即上面排除的情况为，当前状态为0（停止），下一状态为1（行走），同样也不符合要求。
                由于上面已经把当前状态0，下一个状态1的情况剔除，对于某一个棋子，目前还剩下以下三种情况：
                当前状态1，下一状态1，与 = 1 
                当前状态1，下一状态0，与 = 0
                当前状态0，下一状态0，与 = 0
                """
                # new_mask = stopped_mask ^ active
                new_mask = stopped_mask & active

                # calculate new position for role i
                for i in range(len(new_pos)):
                    new_pos[i] = (new_pos[i][0] + dirs[i][0] * ((new_mask >> i) & 1),
                                  new_pos[i][1] + dirs[i][1] * ((new_mask >> i) & 1))

                # if two roles run into the same position
                if len(Counter(new_pos)) < len(dirs):
                    continue
                # if any index is out of area
                all_c = list(chain(*new_pos))
                if min(all_c) <= 0 or max(all_c) > 8:
                    continue
                # valid move, make next step
                dfs(new_pos, dirs, new_mask)

        # rook: move 4-direction
        # queen: move 4-direction and diagonally
        # bishop: move diagonally
        poss = {}
        poss["rook"] = ((1, 0), (-1, 0), (0, 1), (0, -1))
        poss["bishop"] = ((1, 1), (1, -1), (-1, 1), (-1, -1))
        poss["queen"] = ((1, 0), (-1, 0), (0, 1), (0, -1),
                         (1, 1), (1, -1), (-1, 1), (-1, -1))
        # find all dir combinations
        for dirs in product(*(poss[i] for i in pieces)):
            dfs(positions, dirs, (1 << len(pieces)) - 1)  # 10000 - 1 = 1111
        return len(ans)

    def minStartValue(self, nums: List[int]) -> int:
        curr = 0
        ans = 1
        for i in nums:
            curr += i
            # curr should > 0
            ans = max(1 - curr, ans)
        return ans

    def removeElements(self, head: ListNode, val: int) -> ListNode:
        # recursion
        # def check(curr: ListNode):
        #     if not curr: return None
        #     if curr.val == val: return check(curr.next)
        #     else:
        #         curr.next = check(curr.next)
        #         return curr
        # return check(head)
        pre_ans = pre = ListNode(-1)
        pre.next = curr = head
        while curr:
            if curr.val == val:
                pre.next = curr = curr.next
            else:
                pre, curr = curr, curr.next
        return pre_ans.next

    def climbStairs(self, n: int) -> int:
        # dp
        pre_2, pre_1 = 0, 1
        for _ in range(1, n + 1):
            curr = pre_2 + pre_1
            pre_2, pre_1 = pre_1, curr
        return pre_1

    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        sub = dict()
        for n in sorted(nums):
            temp = [n]
            for key, value in sub.items():
                if n % key == 0 and len(value) + 1 > len(temp):
                    # form a new subset
                    temp = value + [n]
            sub[n] = temp
        return sorted(sub.values(), key=lambda x: len(x))[-1]

    def findKthNumber(self, m: int, n: int, k: int) -> int:
        def count(x):
            return sum(min(x // i, n) for i in range(1, m + 1))

        L, R, mid, ans = 0, m * n, 0, 0
        while L <= R:
            mid = (L + R) >> 1
            if count(mid) < k:
                L = mid + 1
            else:
                R, ans = mid - 1, mid
        return ans

    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        # without extra space and run in O(n) time
        nums.append(len(nums) + 1)
        nums.sort()
        next_match = 1
        ans = list()
        for i in nums:
            if i < next_match:
                continue
            if i > next_match:
                for tmp in range(next_match, i):
                    ans.append(tmp)
            next_match = i + 1
        return ans

    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root:
            return None
        if root.val == key:
            if not root.left:
                return root.right
            if not root.right:
                return root.left
            left_r = root.left
            while left_r.right:
                left_r = left_r.right
            left_r.right = root.right
            return root.left
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        else:
            root.right = self.deleteNode(root.right, key)
        return root

    def largestComponentSize(self, nums: List[int]) -> int:
        # union find
        n = max(nums)
        count = Counter()
        group = [i for i in range(n + 1)]

        def find(x):
            if group[x] == x:
                return x
            """
            this line is fucking essential!!! 
            improved time complexity a lot. 
            when you try to find root of x, it takes only 1 step to reach the end.
            """
            group[x] = find(group[x])
            return group[x]

        def union(x, y):
            x_group, y_group = find(x), find(y)
            if x_group != y_group:
                group[x_group] = y_group

        for a in nums:
            for num in range(2, int(math.sqrt(a) + 1)):
                if a % num == 0:
                    union(a, num)
                    union(a, a // num)

        for a in nums:
            count[find(a)] += 1
        return max(count.values())

    def searchInsert(self, nums: List[int], target: int) -> int:
        if target > nums[-1]:
            return len(nums)
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) >> 1
            if nums[mid] == target:
                return mid
            if nums[mid] > target:
                right = mid
            else:
                left = mid + 1
        return left

    def maxSubArray(self, nums: List[int]) -> int:
        # pre_value = pre_posi = pre_nega = 0
        # ans = overall_max= -float('inf')
        # for i in nums:
        #     if i < 0:
        #         if pre_value >= 0: ans = max(pre_posi, ans) # +++-
        #         pre_nega += i
        #     else:
        #         if pre_value < 0:
        #             # ---+
        #             pre_posi = max(pre_posi + pre_nega, 0)
        #             pre_nega = 0
        #         pre_posi += i
        #     pre_value = i
        #     overall_max = max(overall_max, i)
        # ans = max(pre_posi, ans)
        # if overall_max < 0 and ans == 0: ans = overall_max
        # return ans

        # dp
        ans = nums[0]
        pre_min_sum = curr_sum = 0
        for i in nums:
            curr_sum += i
            ans = max(ans, curr_sum - pre_min_sum)
            pre_min_sum = min(pre_min_sum, curr_sum)
        return ans

    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        # union find
        # first: we need to find hiding edges from input
        # if an email exists in both account, then there should be an edge between these two accounts
        n = len(accounts)
        graph = [i for i in range(n)]  # union find root idx list

        def find(x):
            if graph[x] != x:
                graph[x] = find(graph[x])
            return graph[x]

        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                graph[root_x] = root_y

        email_account = dict()  # key = email | value = account list
        for idx, email in enumerate(accounts):
            for e in email[1:]:
                if e in email_account:
                    email_account[e].append(idx)
                else:
                    email_account[e] = [idx]
        for value in email_account.values():
            # if there are more than 1 account share the same email, connect these accounts
            # value = [0, 1, 2], zip(value, value[1:]) = [[0, 1], [1, 2]]
            if len(value) > 1:
                [union(i, j) for i, j in zip(value, value[1:])]

        # key = root account index | value: a set of all emails
        root_email = collections.defaultdict(set)
        for account_idx, root in enumerate(graph):
            root_email[find(root)].update(accounts[account_idx][1:])

        return [accounts[root_idx][0:1] + list(sorted(email_list)) for root_idx, email_list in root_email.items()]

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def find(first: bool):
            left, right, visited = 0, len(nums), False
            while left < right:
                mid = (left + right) >> 1
                if nums[mid] == target:
                    visited = True
                    if first:
                        right = mid
                    else:
                        left = mid + 1
                elif nums[mid] > target:
                    right = mid
                elif nums[mid] < target:
                    left = mid + 1
            if not visited:
                return -1
            return left if first else left - 1
        return [find(True), find(False)]

    def search(self, nums: List[int], target: int) -> int:
        # left, right = 0, len(nums) - 1
        # while left < right:
        #     mid = (left + right) >> 1
        #     if nums[left] == target: return left
        #     if nums[mid] == target: return mid
        #     if nums[right] == target: return right
        #     if nums[mid] > target:
        #         if target < nums[left] < nums[mid]: left = mid + 1
        #         else: right = mid
        #     else:
        #         if nums[mid] < nums[right] < target: right = mid
        #         else: left = mid + 1
        # return left if nums[left] == target else -1

        left, right = 0, len(nums)
        while left < right:
            mid = (left + right) >> 1
            temp = nums[mid]
            if (nums[mid] < nums[0]) != (target < nums[0]):  # not on the same side
                temp = float('inf') if target >= nums[0] else -float('inf')
            if temp == target:
                return mid
            if temp > target:
                right = mid
            else:
                left = mid + 1
        return -1

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # time complexity: O(m + log(n)) | space complexity: O(1)
        for row in matrix:
            if row[0] > target:
                return False
            if row[-1] < target:
                continue
            left, right = 0, len(row)
            while left < right:
                mid = (left + right) >> 1
                if row[mid] == target:
                    return True
                if row[mid] > target:
                    right = mid
                else:
                    left = mid + 1
        return False

    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix:
            return 0
        m, n = len(matrix), len(matrix[0])
        # for each cell, we need:
        # height of 1's
        # index of first 0 from matrix[i][j] to left
        # index of first 0 from matrix[i][j] to right
        height_left_right = [[[0] * 3 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            lm, rm = -1, n
            for j in range(n):
                if matrix[i][j] == '0':
                    lm = j
                    continue
                # set height
                height_left_right[i][j][0] = 1 if i == 0 else height_left_right[i - 1][j][0] + 1
                # set left_most
                height_left_right[i][j][1] = lm
            for j in range(n - 1, -1, -1):
                if matrix[i][j] == '0':
                    rm = j
                else:
                    height_left_right[i][j][2] = rm  # set right_most
        ans = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '0':
                    continue
                hm, lm, rm = height_left_right[i][j]
                for row in range(i + 1 - hm, i):
                    lm = max(lm, height_left_right[row][j][1])
                    rm = min(rm, height_left_right[row][j][2])
                ans = max(ans, hm * (rm - lm - 1))
        return ans

    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) >> 1
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        return nums[left]

    def findPeakElement(self, nums: List[int]) -> int:
        # O(logN) time complexity
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) >> 1
            mid_ = mid + 1
            if nums[mid] > nums[mid_]:
                right = mid
            else:
                left = mid_
        return left

    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:
            return head
        pre = ListNode(-1)
        pre.next = curr = head
        head = pre

        while curr:
            ne = curr.next
            if ne and curr.val == ne.val:
                temp = curr.val
                while curr and curr.val == temp:
                    curr = curr.next
                pre.next = curr
            else:
                pre = curr
                curr = curr.next
        return head.next

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # idx_nums = sorted(enumerate(nums), key=lambda x: x[1])
        nums.sort()
        ans = list()
        for i in range(len(nums)):
            # prun if this number is the same as the previous one
            # avoid containing duplicate triplets
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            target = 0 - nums[i]
            left, right = i + 1, len(nums) - 1
            while left < right:
                temp = nums[left] + nums[right]
                if temp == target:
                    ans.append([nums[i], nums[left], nums[right]])
                    left_visited = nums[left]
                    while left < len(nums) and nums[left] == left_visited:
                        left += 1
                elif temp > target:
                    right -= 1
                else:
                    left += 1
        return ans

    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        odd = head
        even = even_head = head.next
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        odd.next = even_head
        return head

    def backspaceCompare(self, s: str, t: str) -> bool:
        def backspace(st: str) -> str:
            char_list = list()
            for c in st:
                if c == '#':
                    if char_list:
                        char_list.pop()
                else:
                    char_list.append(c)
            return ''.join(char_list)
        return backspace(s) == backspace(t)

    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        first_idx = second_idx = 0
        m, n = len(firstList), len(secondList)
        ans = list()
        while first_idx < m and second_idx < n:
            f_start, f_end = firstList[first_idx]
            s_start, s_end = secondList[second_idx]
            if f_end < s_start:
                first_idx += 1
            elif s_end < f_start:
                second_idx += 1
            else:
                ans.append([max(f_start, s_start), min(f_end, s_end)])
                if f_end <= s_end:
                    first_idx += 1
                if f_end >= s_end:
                    second_idx += 1
        return ans

    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        ans = 0
        while left < right:
            ans = max(ans, min(height[left], height[right]) * (right - left))
            if height[left] > height[right]:
                right -= 1
            else:
                left += 1
        return ans

    def maxProduct(self, nums: List[int]) -> int:
        # # avoid incorrect answer if a negative number is the only element in input list
        # if len(nums) == 1: return nums[0]
        # # dp[i] = [max_prod positive, max_prod negative]
        # # dp[i] represents for the max product of subarray ends in nums[i]
        # dp = [[max(0, nums[0]), min(0, nums[0])]]
        # for i in nums[1:]:
        #     pre_posi, pre_nega = dp[-1]
        #     if i > 0: dp.append([max(pre_posi * i, i), pre_nega * i]) # positive number
        #     elif i < 0: dp.append([pre_nega * i, min(pre_posi * i, i)]) # negative
        #     else: dp.append([0, 0]) # zero
        # return max([x for x, _ in dp])

        # dp does not have to be a list, since we only care about the previous one
        # avoid incorrect answer if a negative number is the only element in input list
        if len(nums) == 1:
            return nums[0]
        # dp = [max_prod positive, max_prod negative]
        dp = [max(0, nums[0]), min(0, nums[0])]
        ans = dp[0]
        for i in nums[1:]:
            pre_posi, pre_nega = dp[0], dp[1]
            if i > 0:
                dp = [max(pre_posi * i, i), pre_nega * i]  # positive number
            elif i < 0:
                dp = [pre_nega * i, min(pre_posi * i, i)]  # negative
            else:
                dp = [0, 0]  # zero
            ans = max(ans, dp[0])
        return ans

    def findAnagrams(self, s: str, p: str) -> List[int]:
        n = len(p)
        char_count = collections.Counter(s[:n])
        char_count_p = collections.Counter(p)
        ans = list()
        if char_count == char_count_p:
            ans.append(0)

        for i in range(1, len(s) - n + 1):
            char_count[s[i - 1]] -= 1
            if char_count[s[i - 1]] == 0:
                char_count.pop(s[i - 1])
            char_count[s[i + n - 1]] += 1
            if char_count == char_count_p:
                ans.append(i)
        return ans

    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        # sliding windows
        start = end = ans = 0
        prod = 1
        while end < len(nums):
            prod *= nums[end]
            while start <= end and prod >= k:
                prod /= nums[start]
                start += 1
            # add all subarrays end with nums[end]
            ans += end - start + 1
            end += 1
        return ans

    def minCostToMoveChips(self, position: List[int]) -> int:
        ans = [0] * 2  # even, odd
        for i in position:
            ans[i % 2] += 1
        return min(ans)

    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        n = len(nums)
        left = right = curr_sum = 0
        ans = float('inf')
        while right < n:
            curr_sum += nums[right]
            right += 1
            if curr_sum >= target:
                while curr_sum >= target:
                    curr_sum -= nums[left]
                    left += 1
                ans = min(ans, right - left + 1)
        return ans if ans != float('inf') else 0

    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if grid[0][0] == 1 or grid[n - 1][n - 1] == 1:
            return -1
        dir = [[0, 1], [0, -1], [1, 0], [-1, 0],
               [1, 1], [1, -1], [-1, 1], [-1, -1]]
        visited = set()
        curr_list = [[n - 1, n - 1, 1]]  # element: [x_idx, y_idx, step]
        while curr_list:
            x, y, step = curr_list.pop(0)
            if x == 0 and y == 0:
                return step
            for i, j in dir:
                x_, y_ = x + i, y + j
                if n > x_ >= 0 <= y_ < n and grid[x_][y_] == 0 and (x_, y_) not in visited:
                    visited.add((x_, y_))
                    curr_list.append([x_, y_, step + 1])
        return -1

    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        m, n = len(board), len(board[0])
        dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        # flip_set = set()
        # def capture(x, y) -> bool:
        #     flip_set.add((x, y))
        #     for i, j in dir:
        #         x_, y_ = x + i, y + j
        #         if x_ == m or x_ == -1 or y_ == n or y_ == -1: return False
        #         if board[x_][y_] == 'O' and (x_, y_) not in flip_set:
        #             temp = capture(x_, y_)
        #             if not temp: return False
        #     return True

        # for i in range(m):
        #     for j in range(n):
        #         flip_set.clear()
        #         if board[i][j] == 'O' and capture(i, j):
        #             for ii, jj in flip_set:
        #                 board[ii][jj] = 'X'

        boarder_o = set()
        for i in range(m + n):
            boarder_o |= {(0, i), (m - 1, i), (i, 0), (i, n - 1)}

        while boarder_o:
            i, j = boarder_o.pop()
            if m > i >= 0 <= j < n and board[i][j] == 'O':
                board[i][j] = 'S'
                [boarder_o.add((i + ii, j + jj)) for ii, jj in dir]

        board[:] = [['XO'[i == 'S'] for i in row] for row in board]

    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        # DFS
        n = len(graph)
        ans = list()

        def move_next(curr, path):
            if curr == n - 1:
                ans.append(path.copy())
                return
            for node in graph[curr]:
                path.append(node)
                move_next(node, path)
                path.pop()

        move_next(0, [0])
        return ans

    def getDecimalValue(self, head: ListNode) -> int:
        ans = 0
        while head:
            ans *= 2
            ans += head.val
            head = head.next
        return ans

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        ans = set()
        ans.add(tuple())
        for i in nums:
            new = set()
            for j in ans:
                temp = list(j)
                temp.append(i)
                new.add(tuple(temp))
            ans |= new
        return [list(i) for i in ans]

    def subsets(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        ans = list()
        ans.append([])
        for i in nums:
            n = len(ans)
            for j in range(n):
                ans.append(ans[j] + [i])
        return ans

    def findTilt(self, root: TreeNode) -> int:
        global ans
        ans = 0

        def count_sum(node: TreeNode):
            if not node:
                return 0
            global ans
            left, right = count_sum(node.left), count_sum(node.right)
            ret = left + right + node.val
            ans += abs(left - right)
            return ret
        count_sum(root)
        return ans

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        count = collections.Counter(nums)
        ans = list()

        def find_next(li: list):
            if len(li) == len(nums):
                ans.append(li.copy())
                return
            for key in count.keys():
                if count[key] > 0:
                    li.append(key)
                    count[key] -= 1
                    find_next(li)
                    li.pop()
                    count[key] += 1
        find_next([])
        return ans

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ans = list()

        def find_next(pre_sum: int, idx: int, li: list):
            # to avoid making duplicate combinations
            # we cannot choose a number that has a less idx than the previous one
            # which means, our combinations are formed in non-descending idx order
            if pre_sum == target:
                ans.append(li.copy())
                return
            for i in range(idx, len(candidates)):
                if pre_sum + candidates[i] <= target:
                    li.append(candidates[i])
                    find_next(pre_sum + candidates[i], i, li)
                    li.pop()
        find_next(0, 0, [])
        return ans

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        ans = list()
        count = collections.Counter(candidates)

        def find_next(pre_sum: int, li: list):
            if pre_sum == target:
                ans.append(li.copy())
                return
            for key in sorted(count.keys()):
                if pre_sum + key > target:
                    break
                if li and key < li[-1]:
                    continue
                if count[key] <= 0:
                    continue
                li.append(key)
                count[key] -= 1
                find_next(pre_sum + key, li)
                li.pop()
                count[key] += 1
        find_next(0, [])
        return ans

    def maxPower(self, s: str) -> int:
        ans = curr = 1
        pre = s[0]
        for c in s[1:]:
            if c == pre:
                curr += 1
                ans = max(ans, curr)
            else:
                curr = 1
            pre = c
        return ans

    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        digit_2_letter = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }

        ans = list()

        def append_next(idx, pre):
            if idx == len(digits):
                ans.append(pre)
            else:
                [append_next(idx + 1, pre + l)
                 for l in digit_2_letter[digits[idx]]]
        append_next(0, '')
        return ans

    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        def sum_sub(node: TreeNode):
            if not node:
                return 0
            if node.val < low:
                return sum_sub(node.right)
            if node.val > high:
                return sum_sub(node.left)
            return node.val + sum_sub(node.left) + sum_sub(node.right)
        return sum_sub(root)

    def generateParenthesis(self, n: int) -> List[str]:
        # stack
        ans = list()

        def find_next(pre: int, curr: str, remain: int):
            if remain < 0 or pre < 0:
                return
            if len(curr) == n * 2:
                ans.append(curr)
            find_next(pre + 1, curr + '(', remain - 1)  # '('
            find_next(pre - 1, curr + ')', remain)  # ')'

        find_next(0, '', n)
        return ans

    def exist(self, board: List[List[str]], word: str) -> bool:
        # DFS
        m, n = len(board), len(board[0])
        if len(word) > m * n:
            return False
        dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        def find_next(x: int, y: int, w: str, visited: set):
            if not w:
                return True
            for i, j in dir:
                x_, y_ = x + i, y + j
                if m > x_ >= 0 <= y_ < n and board[x_][y_] == w[0] and (x_, y_) not in visited:
                    if find_next(x_, y_, w[1:], visited | {(x_, y_)}):
                        return True
            return False

        return any(find_next(i, j, word[1:], {(i, j)}) for i in range(m) for j in range(n) if board[i][j] == word[0])

    def uniquePaths(self, m: int, n: int) -> int:
        # dp
        dp = [1] * n
        for _ in range(1, m):
            for i in range(1, n):
                dp[i] += dp[i - 1]
        return dp[-1]

    def longestPalindrome(self, s: str) -> str:
        # dp[i][j] represents for the longest palindrome of substring s[i:j]
        n = len(s)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n):
            dp[i][i + 1] = 1
        for step in range(2, n + 1):
            for left in range(n + 1 - step):
                if s[left] == s[left + step - 1] and (step == 2 or dp[left + 1][left + step - 1] > 0):
                    dp[left][left + step] = 2 + dp[left + 1][left + step - 1]
        ans = ''
        print(dp)
        for i in range(n + 1):
            for j in range(n + 1):
                if dp[i][j] > len(ans):
                    ans = s[i:j]
        return ans

    def maximalSquare(self, matrix: List[List[str]]) -> int:
        # square, much easier
        # dp
        dp = [int(i) for i in matrix[0]]
        ans = max(dp)
        for row in matrix[1:]:
            temp = [int(row[0])]
            for idx in range(1, len(row)):
                temp.append(0 if row[idx] == '0' else 1 +
                            min([temp[-1], dp[idx - 1], dp[idx]]))
            ans = max(ans, max(temp))
            dp = temp
        return ans ** 2

    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        # contiguous
        if len(nums) < 3:
            return 0
        ans = 0
        curr = [nums[0]]
        for i in nums[1:]:
            if len(curr) == 1:
                curr.append(i)
            else:
                if i - curr[-1] == curr[-1] - curr[-2]:
                    curr.append(i)
                    ans += max(0, len(curr) - 2)
                else:
                    curr = [curr[-1], i]
        return ans

    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        divider, ans = 1, 0
        while divider <= left:
            if left // divider == right // divider and left & divider != 0:
                ans |= divider
            divider <<= 1
        return ans

        # into one-line
        return sum([2 ** expo for expo in range(math.floor(math.log2(left) + 1)) if left // (2 ** expo) == right // (2 ** expo) and left & (2 ** expo) != 0]) if left > 0 else 0

    def numDecodings(self, s: str) -> int:
        # dp = [total ways, ways end with 1, ways end with 2, ways end with 1 digit]
        dp = [1, 0, 0, 0]
        for c in s:
            curr = int(c)
            dp = [(dp[0] if curr != 0 else 0) + dp[1] + (dp[2] if curr < 7 else 0), dp[0]
                  if curr == 1 else 0, dp[0] if curr == 2 else 0, dp[0] if curr != 0 else 0]
        return dp[0]

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        tr = Trie()
        [tr.insert(word) for word in wordDict]

        @functools.cache
        def find_word(ss: str):
            if not ss:
                return True
            return any(tr.search(ss[:i + 1]) and find_word(ss[i + 1:]) for i in range(len(ss)))

        return find_word(s)

    def decodeString(self, s: str) -> str:
        stack_str = list()
        stack_number = list()
        curr = 0
        ans = ''
        for c in s:
            if c.isalpha():
                if stack_str:
                    stack_str[-1] += c
                else:
                    ans += c
            elif c.isdigit():
                curr = curr * 10 + int(c)
            elif c == '[':
                stack_str.append('')
                stack_number.append(curr)
                curr = 0
            elif c == ']':
                t_str = stack_str.pop()
                t_number = stack_number.pop()
                if stack_str:
                    stack_str[-1] += t_str * t_number
                else:
                    ans += t_str * t_number
        return ans

    def isPowerOfTwo(self, n: int) -> bool:
        while n > 2:
            if n & 1:
                return False
            n >>= 1
        return n > 0

    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        n, head_t, tail = 0, head, None
        while head_t:
            n += 1
            temp = ListNode(head_t.val)
            temp.next = tail
            tail = temp
            head_t = head_t.next
        # now we have head & tail, with total amount n
        ans = head
        tail_next, head_next = tail, head
        for _ in range(n >> 1):
            tail = tail_next
            head_next = head.next
            tail_next = tail.next
            head.next = tail
            tail.next = head_next
            head = head_next
        if n & 1:
            head.next = None
        else:
            tail.next = None
        return ans

    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        prev = collections.defaultdict(set)
        follow = collections.defaultdict(set)
        for curr, pre in prerequisites:
            prev[curr].add(pre)
            follow[pre].add(curr)
        next_course = [i for i in range(numCourses) if i not in prev]
        ans = list()
        while next_course:
            temp = list()
            for i in next_course:
                for f in follow[i]:
                    prev[f].remove(i)
                    if not prev[f]:
                        temp.append(f)
            ans += next_course
            next_course = temp
        return ans if len(ans) == numCourses else []

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        ans = list()
        pre_s, pre_e = intervals[0]
        for start, end in intervals[1:]:
            if pre_e >= start:
                pre_e = max(pre_e, end)
            else:
                ans.append([pre_s, pre_e])
                pre_s, pre_e = start, end
        return ans + [[pre_s, pre_e]]

    def findComplement(self, num: int) -> int:
        return 2 ** (num.bit_length()) - num - 1 if n > 0 else 1

    def fractionAddition(self, expression: str) -> str:
        # if expression[0] != '-': expression = '+' + expression
        # m = re.findall('[+-][\d]+\/[\d]+', expression)
        # sub = list()
        # overall_lcm = 1
        # def gcd(x, y):
        #     while y: x, y = y, x % y
        #     return x

        # def lcm(x, y):
        #     return abs(x * y) // gcd(x, y)

        # for i in m:
        #     fraction = list(map(int, i[1:].split('/')))
        #     sub.append([i[0], fraction[0], fraction[1]])
        #     overall_lcm = lcm(overall_lcm, fraction[1])
        # print(sub)
        # for i in range(len(sub)):
        #     sub[i][1] *= overall_lcm // sub[i][2]
        # print(sub)
        # overall_nomi = sum([-nomi if sign == '-' else nomi for sign, nomi, _ in sub])
        # ans = '-' if overall_nomi < 0 else ''
        # overall_nomi = abs(overall_nomi)
        # g = gcd(overall_lcm, overall_nomi)
        # return ans + f'{overall_nomi // g}/{overall_lcm // g}'

        # too complicated
        m = map(int, re.findall('[+-]?[\d]+', expression))
        nomi, deno = 0, 1  # initialize result to 0/1
        for i in m:
            n = next(m)
            nomi = nomi * n + i * deno
            deno *= n
            g = math.gcd(nomi, deno)
            nomi //= g
            deno //= g
        return f'{nomi}/{deno}'

    def buildArray(self, nums: List[int]) -> List[int]:
        # return [nums[nums[i]] for i in range(len(nums))]
        n = len(nums)
        for i in range(n):
            nums[i] += n * (nums[nums[i]] % n)
        for i in range(n):
            nums[i] //= n
        return nums

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

    def smallestRepunitDivByK(self, k: int) -> int:
        """
        pos: 1, 3, 7, 9
        imp: 2, 4, 5, 6, 8, 0
        """
        if k % 10 in [2, 4, 5, 6, 8, 0]:
            return -1
        visited = set()
        amount = 0
        curr_mod = 0
        while True:
            amount += 1
            curr_mod = (curr_mod * 10 + 1) % k
            if curr_mod == 0:
                return amount
            if curr_mod in visited:
                return -1
            visited.add(curr_mod)

    def getOrder(self, tasks: List[List[int]]) -> List[int]:
        # heapq element: (process time, index)
        pq = []
        tasks = sorted([[task[0], task[1], index]
                       for index, task in enumerate(tasks)])
        ans = list()
        curr_time = 0
        while tasks or pq:
            # if heapq is empty, we must add the first tasks into it, and update curr_time if needed
            if not pq:
                curr_time = max(tasks[0][0], curr_time)
            while tasks and tasks[0][0] <= curr_time:
                temp = tasks.pop(0)
                heapq.heappush(pq, (temp[1], temp[2]))
            # pq is not empty now
            # pop out the first task
            process, index = heapq.heappop(pq)
            curr_time += process - 1
            ans.append(index)
        return ans

    def canCross(self, stones: List[int]) -> bool:
        # recursion
        # visited = set()
        # def jump(curr, k):
        #     if (curr, k) in visited: return False
        #     visited.add((curr, k))
        #     if curr not in stones: return False
        #     if curr == stones[-1]: return True
        #     # curr position in stones, jump to next place
        #     step = [k - 1, k, k + 1]
        #     return any(jump(curr + i, i) for i in step)
        # return jump(1, 1) if stones[1] == 1 else False

        # dp
        n = len(stones)
        dp = [[0] * (n + 1) for _ in range(n)]
        dp[0][1] = 1  # frog can only jump 1 step at stones[0] = 0
        for i in range(1, n):
            for j in range(i - 1, -1, -1):  # from right to left to allow us prune iteration
                step = stones[i] - stones[j]
                if step > j + 1:
                    break  # frog can jump at most j + 1 step at stones[j]
                if dp[j][step]:
                    dp[i][step - 1] = dp[i][step] = dp[i][step + 1] = 1
        return any(dp[-1])

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
                if buy[0][1] == 0:
                    heapq.heappop(buy)
                if sell[0][1] == 0:
                    heapq.heappop(sell)
        return sum(amount for _, amount in buy + sell) % (10 ** 9 + 7)

    def maxAncestorDiff(self, root: TreeNode) -> int:

        def find_diff(node: TreeNode, min_anc: int, max_anc: int):
            if not node:
                return -1
            diff = [max(abs(node.val - min_anc), abs(node.val - max_anc))]
            min_anc = min(node.val, min_anc)
            max_anc = max(node.val, max_anc)
            diff.append(find_diff(node.left, min_anc, max_anc))
            diff.append(find_diff(node.right, min_anc, max_anc))
            return max(diff)

        return max(find_diff(root.left, root.val, root.val), find_diff(root.right, root.val, root.val))

    def insertionSortList(self, head: ListNode) -> ListNode:
        pre = ListNode(-1)
        pre.next = to_insert = head
        while head and head.next:
            if head.val > head.next.val:
                to_insert = ListNode(head.next.val)
                head.next = head.next.next
                # remove head.next and insert it somewhere else
                # point head.next to head.next.next
                find_pos = pre
                while find_pos.next.val < to_insert.val:
                    find_pos = find_pos.next
                # insert between find_post and find_pos.next
                temp = find_pos.next
                find_pos.next, to_insert.next = to_insert, temp
            else:
                head = head.next
        return pre.next

    def maximumInvitations(self, favorite: List[int]) -> int:
        # case 1: 寻找最大的环， size >= 3
        # case 2: 找到所有互相喜欢的员工（pairs），以两者为起点，反向延伸找到最长的被喜欢链len = a & b
        # sum(a + b + 2 for all pairs)
        # return max(case 1, case 2)

        n = len(favorite)
        cycles = list()

        # form a be-liked dict to search employee backwards
        liked = collections.defaultdict(list)
        for i, j in enumerate(favorite):
            liked[j].append(i)

        # case 1
        for i in range(n):
            if favorite[i] == -1:
                continue
            path = {i: 0}
            while favorite[i] != -1:
                temp = favorite[i]
                favorite[i] = -1
                if temp in path:
                    cycles.append([temp, len(path) - path[temp]])
                    break
                i = temp
                path[temp] = len(path)

        # case 2
        # we already detect all pairs in case 1
        def extend(idx, exclude):
            pre = liked[idx]
            ans = 0
            for pre in liked[idx]:
                if pre != exclude:
                    ans = max(ans, extend(pre, exclude))
            return ans + 1

        max_case1, max_case2 = 0, 0
        for i, size in cycles:
            if size > 2:
                max_case1 = max(max_case1, size)
            else:
                j = [temp for temp in liked[i] if i in liked[temp]][0]
                max_case2 += extend(i, j) + extend(j, i)
        return max(max_case1, max_case2)

        # 寻找最大的环
        global ans
        ans = 0
        n = len(favorite)
        visited = set()

        # pair = set()
        # for i in range(len(favorite)):
        #     if favorite[favorite[i]] == i:
        #         pair.add((i, favorite[i]))

        def invite(idx: int, pre: list):
            # print(idx, pre)
            global ans
            inv = favorite[idx]
            if inv not in pre:
                pre.append(inv)
                invite(inv, pre)
            else:
                for e in pre:
                    visited.add(e)
                start_index = pre.index(inv)
                ans = max(ans, len(pre) - start_index)
                print(pre, start_index, len(pre) - start_index)
                if inv == pre[-2]:
                    # for x, y in pair:
                    #     if x not in pre and y not in pre:
                    #         pre += [x, y]
                    ans = max(ans, len(pre))
                    for i in range(n):
                        if i not in pre:
                            invite(i, pre + [i])
        for i in range(n):
            if i not in visited:
                invite(i, [i])
        return ans
        
    def catMouseGame(self, graph: List[List[int]]) -> int:
        # [1,0,0,2,1,4,7,8,9,6,7,10,8]
        # 有向有环图
        n = len(graph)

        @functools.cache
        def move(step, m, c):
            """
            step: even: mouse turn | odd: cat turn
            m: mouse's position
            c: cat's position
            """
            if step == 2 * n:
                return 0  # there is no winner after 2n steps, then they will end up draw
            if m == c:
                return 2  # mouse and cat are in the same position, cat wins
            if m == 0:
                return 1  # mouse reaches hole, mouse wins

            # move next step
            if step % 2 == 0:
                # mouse turn
                # mouse will take the step optimally
                # once mouse find a chance to win, it will take this step
                if any(move(step + 1, nxt, c) == 1 for nxt in graph[m]):
                    return 1
                # if there is no chance to win, mouse will look for the draw
                if any(move(step + 1, nxt, c) == 0 for nxt in graph[m]):
                    return 0
                # if there is no chance to end with either mouse win or draw, then cat will win
                return 2
            else:
                # cat turn
                if any(move(step + 1, m, nxt) == 2 for nxt in graph[c] if nxt != 0):
                    return 2
                if any(move(step + 1, m, nxt) == 0 for nxt in graph[c] if nxt != 0):
                    return 0
                return 1
        return move(0, 1, 2)  # game start with mouse at 1 and cat at 2

    def canMouseWin(self, grid: List[str], catJump: int, mouseJump: int) -> bool:
        # allowed steps:
        # 4-directionally, less than or equal to the maximum jump step, without cross the wall
        m, n = len(grid), len(grid[0])
        dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        mx = my = cx = cy = position = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] != '#':
                    position += 1
                if grid[i][j] == 'M':
                    mx, my = i, j
                if grid[i][j] == 'C':
                    cx, cy = i, j

        @functools.cache
        def find_nxt(x, y, jump):
            ans = list()
            for dx, dy in dir:
                for i in range(jump + 1):
                    nx, ny = x + dx * i, y + dy * i
                    if m > nx >= 0 <= ny < n and grid[nx][ny] != '#':
                        ans.append([nx, ny])
                    else:
                        break
            return ans

        @functools.cache
        def move(step, mx, my, cx, cy):
            if step > position * 2:
                return False
            if mx == cx and my == cy:
                return False
            if grid[mx][my] == 'F':
                return True
            if grid[cx][cy] == 'F':
                return False

            if step % 2 == 0:
                # mouse turn
                return True if any(move(step + 1, x, y, cx, cy) for x, y in find_nxt(mx, my, mouseJump)) else False
            else:
                # cat turn
                return False if any(not move(step + 1, mx, my, x, y) for x, y in find_nxt(cx, cy, catJump)) else True

        return move(0, mx, my, cx, cy)

    def busiestServers(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
        # two-heap
        # ans[i] represents for the amount of requests ith server handled
        ans = [0] * k
        # heap: servers that currently occupied by a request, element = (free_time, idx)
        busy = []
        # heap: servers that currently free to handle request, element = (idx)
        free = [i for i in range(k)]
        for idx, (start, last) in enumerate(zip(arrival, load)):
            while busy and busy[0][0] <= start:
                _, i = heapq.heappop(busy)
                # instead of push i into free, we record idx + (i - idx) % k here as the server index
                # believe it or not, this magic expression ensure we sort server in idx % k -> k - 1 -> 0 -> idx % k - 1 order
                """ i: index of server | idx: index of request
                i               expression                      record in heap free         sort-order | range(idx, idx + k)
                idx % k         idx + (i - idx) % k             idx                         idx + k - k
                idx % k + 1                                     idx + 1                     idx + k - (k - 1)
                k - 1           idx + (- 1 - idx) % k           idx + k - idx % k - 1       idx + k - (idx % k + 1)
                k               idx + (- idx) % k               idx + k - idx % k           idx + k - (idx % k)
                idx % k - 1     idx + (idx % k - 1 - idx) % k   idx + (- 1) % k             idx + k - 1
                """
                heapq.heappush(free, idx + (i - idx) % k)
            if free:
                assign = heapq.heappop(free) % k
                heapq.heappush(busy, (start + last, assign))
                ans[assign] += 1
        most = max(ans)
        return [i for i, cnt in enumerate(ans) if cnt == most]

        # three-heap
        # ans[i] represents for the amount of requests ith server handled
        ans = [0] * k
        # heap: servers that currently occupied by a request, element = (free_time, idx)
        busy = []
        # heap: servers that currently free to handle request, idx equal or greater than i, element = (idx)
        free_behind = []
        # heap: servers that currently free to handle request, idx less than i
        free_ahead = [i for i in range(k)]
        for idx, (start, last) in enumerate(zip(arrival, load)):
            idx %= k
            if idx == 0:
                free_behind = free_ahead
                free_ahead = []
            while busy and busy[0][0] <= start:
                _, i = heapq.heappop(busy)
                if i >= idx:
                    heapq.heappush(free_behind, i)
                else:
                    heapq.heappush(free_ahead, i)
            while free_behind and free_behind[0] < idx:
                heapq.heappush(free_ahead, heapq.heappop(free_behind))
            use_heap = free_behind if free_behind else free_ahead
            if use_heap:
                assign = heapq.heappop(use_heap)
                heapq.heappush(busy, (start + last, assign))
                ans[assign] += 1
        most = max(ans)
        return [i for i, cnt in enumerate(ans) if cnt == most]

        # no-heap, TLE
        # ans[i] represents for the amount of requests ith server handled
        ans = [0] * k
        # free_time represents for the time that ith server will be free to take request
        free_time = [-1] * k

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
            if cnt > most:
                most, ret = cnt, [idx]
            elif cnt == most:
                ret.append(idx)
        return ret

    def makeLargestSpecial(self, s: str) -> str:
        start = 0
        cnt = 0
        special = list()
        for i, c in enumerate(s):
            if c == '1':
                cnt += 1
            if c == '0':
                cnt -= 1
            if cnt == 0:
                special.append(
                    '1' + self.makeLargestSpecial(s[start + 1:i]) + '0')
                start = i + 1
                cnt = 0
        return ''.join(sorted(special)[::-1])

    def partition(self, s: str) -> List[List[str]]:
        # dp[i][j] = True if s[i:j] is a palindrome
        n = len(s)
        dp = [[False] * (n + 1) for _ in range(n + 1)]
        for diff in range(n + 1):
            for i in range(n + 1 - diff):
                j = i + diff
                if diff <= 1:
                    dp[i][j] = True
                elif s[i] == s[j - 1]:
                    dp[i][j] = dp[i + 1][j - 1]

        def find_part(start: int):
            ans = list()
            for i in range(start, n):
                if dp[start][i + 1]:  # is palindrome
                    if i + 1 < n:
                        ans += [[s[start:i + 1]] + p for p in find_part(i + 1)]
                    else:
                        ans.append([s[start:i + 1]])
            return ans
        return find_part(0)

    def checkPartitioning(self, s: str) -> bool:
        # dp[i][j] = True if s[i:j] is a palindrome
        n = len(s)
        dp = [[False] * (n + 1) for _ in range(n + 1)]
        for diff in range(n + 1):
            for i in range(n + 1 - diff):
                j = i + diff
                if diff <= 1:
                    dp[i][j] = True
                elif s[i] == s[j - 1]:
                    dp[i][j] = dp[i + 1][j - 1]

        def part(start: int, cnt: int):
            if start == n:
                return False
            if cnt == 1:
                return dp[start][n]
            return any(part(i + 1, cnt - 1) for i in range(start, n) if dp[start][i + 1])

        return part(0, 3)

    def modifyString(self, s: str) -> str:
        s = '0' + s + '0'
        for i in range(len(s) - 2):
            if s[i + 1] == '?':
                r = ord('a')
                while True:
                    if chr(r) in [s[i], s[i + 2]]:
                        r += 1
                    else:
                        s = s[:i + 1] + chr(r) + s[i + 2:]
                        break
        return s[1:-1]

    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        size = len(primes)
        ugly = [1]  # first ugly number is 1
        # ugly[ugly_idx[i]] is the multiple number we should use for primes[i]
        ugly_idx = [0] * size
        remain = primes.copy()  # or primes.copy()
        for _ in range(1, n):
            # print(len(remain))
            temp = min(remain)
            ugly.append(temp)
            for i in range(size):
                if remain[i] == temp:
                    ugly_idx[i] += 1
                    remain[i] = primes[i] * ugly[ugly_idx[i]]
        return ugly[-1]

    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        # remain = sorted([[fromi, toi, numi] for numi, fromi, toi in trips])
        # travel = []
        # cnt = 0

        # while remain:
        #     fromi = remain[0][0]
        #     while travel and travel[0][0] <= fromi:
        #         _, gone = heapq.heappop(travel)
        #         cnt -= gone
        #     while remain and remain[0][0] <= fromi:
        #         _, toi, numi = heapq.heappop(remain)
        #         heapq.heappush(travel, [toi, numi])
        #         cnt +=  numi
        #     print(cnt, fromi, remain, travel)
        #     if cnt > capacity: return False
        # return True

        for _, change in sorted([x for n, f, t in trips for x in [[f, n], [t, -n]]]):
            capacity -= change
            if capacity < 0:
                return False
        return True

    def checkMove(self, board: List[List[str]], rMove: int, cMove: int, color: str) -> bool:
        dirs = [[0, 1], [0, -1], [1, 0], [-1, 0],
                [1, 1], [1, -1], [-1, 1], [-1, -1]]
        # horizontal | vertical | diagonal

        def extend(x, y, dx, dy, end_color):
            i = 1
            while True:
                nx, ny = x + i * dx, y + i * dy
                if 8 > nx >= 0 <= ny < 8:
                    if board[nx][ny] == '.':
                        return False
                    if board[nx][ny] == end_color:
                        return False if i == 1 else True
                    i += 1
                else:
                    return False

        return any(extend(rMove, cMove, i, j, color) for i, j in dirs)

    def maxSatisfaction(self, satisfaction: List[int]) -> int:
        # iterate in reverse order
        satisfaction = sorted(satisfaction)[::-1]
        ans = total = 0
        for sat in satisfaction:
            total += sat
            if total > 0:
                ans += total
            else:
                break
        return ans

    def evaluate(self, s: str, knowledge: List[List[str]]) -> str:
        # knowledge = dict(knowledge)
        # for key in set(re.findall('\([a-z]+\)', s)):
        #     if key[1:-1] in knowledge:
        #         s = s.replace(key, knowledge[key[1:-1]])
        #     else:
        #         s = s.replace(key, '?')
        # return s

        # another approach
        knowledge = dict(knowledge)
        s = s.split('(')
        for idx, sub in enumerate(s):
            if ')' not in sub:
                continue
            sub = sub.split(')')
            s[idx] = (knowledge[sub[0]] if sub[0]
                      in knowledge else '?') + sub[1]
        return ''.join(s)

    def sumRootToLeaf(self, root: TreeNode) -> int:
        # [1,0,1,0,1,0,1]
        def find_leaf(pre: int, node: TreeNode) -> int:
            pre = pre * 2 + node.val
            if not node.left and not node.right:
                # node is a leaf
                return pre
            return (find_leaf(pre, node.left) if node.left else 0) + (find_leaf(pre, node.right) if node.right else 0)

        return find_leaf(0, root)

    def possibleToStamp(self, grid: List[List[int]], stampHeight: int, stampWidth: int) -> bool:
        m, n = len(grid), len(grid[0])
        H, W = stampHeight, stampWidth

        def acc_2d(grid):
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for c, r in product(range(n), range(m)):
                dp[r + 1][c + 1] = dp[r + 1][c] + \
                    dp[r][c + 1] - dp[r][c] + grid[r][c]
            return dp

        def sumRegion(mat, r1, c1, r2, c2):
            return mat[r2 + 1][c2 + 1] - mat[r1][c2 + 1] - mat[r2 + 1][c1] + mat[r1][c1]

        dp = acc_2d(grid)
        stamp_grid = [[0] * n for _ in range(m)]
        for r, c in product(range(m - H + 1), range(n - W + 1)):
            if sumRegion(dp, r, c, r + H - 1, c + W - 1) == 0:
                # all cells in this range are empty
                # just mark the right-bottom corner cell with 1
                stamp_grid[r + H - 1][c + W - 1] = 1

        stamp_prefix = acc_2d(stamp_grid)
        for r, c in product(range(m), range(n)):
            # cell is empty and cannot be a right-bottom corner of a stamp
            if grid[r][c] == 0 and stamp_grid[r][c] == 0:
                if sumRegion(stamp_prefix, r, c,
                             min(r + H - 1, m - 1),
                             min(c + W - 1, n - 1)) == 0:
                    # this cell cannot be covered by any valid right-bottom corner
                    return False
        return True

    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        """insert a new node into BST

        Args:
            root (TreeNode): BST to insert
            val (int): value of new node

        Returns:
            TreeNode: new BST
        """
        if not root:
            return TreeNode(val)
        if root.val > val:
            root.left = self.insertIntoBST(root.left, val)
        else:
            root.right = self.insertIntoBST(root.right, val)
        return root

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
        digit = [('z', 'zero', 0), ('x', 'six', 6), ('w', 'two', 2), ('u', 'four', 4), ('g', 'eight', 8),
                 ('h', 'three', 3), ('o', 'one', 1), ('f', 'five', 5), ('i', 'nine', 9), ('s', 'seven', 7)]
        digit_cnt = [0] * 10
        for ch, letter, idx in digit:
            temp = cnt[ch]
            for c in letter:
                cnt[c] -= temp
            digit_cnt[idx] = temp
        for idx, count in enumerate(digit_cnt):
            ans += str(idx) * count
        return ans

    def findMinArrowShots(self, points: List[List[int]]) -> int:
        """
        balloons are represented with xstart and xend.
        arrow shot in [xstart, xend] will burst this balloon.

        Args:
            points (List[List[int]]): balloons

        Returns:
            int: minimum number of arrows needed to be shot to burst all balloons
        """

        # find the biggest non-overlap subset
        # 1. form a heap queue(pq) from point, which is sorted by end time.
        # 2. pop the first element from pq, which has the smallest end time. increment answer by 1, and update current end time.
        # 3. if element's start time is less than or equal to current end time, ignore it since it has already been bursted by previous arrows.

        # pq = sorted([[end, start] for start, end in points])
        # curr_end = - 2 ** 31 - 1
        # ans = 0
        # while pq:
        #     end, start = heapq.heappop(pq)
        #     if start > curr_end:
        #         ans += 1
        #         curr_end = end

        # return ans
        # heapq gets worse time complexity???

        points.sort(key=lambda x: x[1])
        index = count = 0
        while index < len(points):
            shot, count = points[index][1], count + 1
            while index < len(points) and points[index][0] <= shot:
                index += 1
        return count

    def dominantIndex(self, nums: List[int]) -> int:
        """determine if the largest element is at least twice as all other elements

        Args:
            nums (List[int]): number list

        Returns:
            int: largest element's index if it satisfies the requirement, or -1 otherwise
        """
        if len(nums) == 1:
            return 0
        nums = sorted([[v, i] for i, v in enumerate(nums)])
        return -1 if nums[-1][0] < nums[-2][0] * 2 else nums[-1][1]

    def minSwaps(self, s: str) -> int:
        n = len(s)
        cnt = collections.Counter(s)
        diff = abs(cnt['0'] - cnt['1'])
        if diff > 1:
            return -1
        if diff == 1:
            if cnt['0'] > cnt['1']:
                template = [chr(ord('0') + i % 2) for i in range(n)]
            else:
                template = [chr(ord('1') - i % 2) for i in range(n)]
        else:  # diff == 0
            template = [chr(ord('0') + i % 2) for i in range(n)]
        ans = sum(i != j for i, j in zip(s, template))
        return ans // 2 if diff == 1 else min(ans, n - ans) // 2

    def myAtoi(self, s: str) -> int:
        res = re.search('\A[\s]*[-+]?[0-9]+', s)
        if not res:
            return 0
        ans = int(res.group())
        return 2 ** 31 - 1 if ans > 2 ** 31 - 1 else -2 ** 31 if ans < -2 ** 31 else ans

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
        # frog is initially on vertex 1, and its possibility is 1/1 after 0 jump
        curr = {1: 1}
        for _ in range(t):
            # frog reached target vertex in less than t jumps
            if target in curr:
                # there is still other vertex to jump, frog will not stuck on target
                if any(c not in visited for c in tree[target]):
                    return 0
                # there is no vertex to jump, frog will stuck on target
                else:
                    return 1 / curr[target]
            temp = dict()
            for node, possibility in curr.items():
                visited.add(node)
                nxt = [c for c in tree[node] if c not in visited]
                for n in nxt:
                    temp[n] = possibility * len(nxt)
            curr = temp
        # frog failed to reach target within t jumps
        return 1 / curr[target] if target in curr else 0

    def wordPattern(self, pattern: str, s: str) -> bool:
        word = dict()
        word_re = dict()
        if len(pattern) != len(s.split()):
            return False
        for i, j in zip(pattern, s.split()):
            if i not in word:
                word[i] = j
            if j not in word_re:
                word_re[j] = i
            if j != word[i] or i != word_re[j]:
                return False
        return True

    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        root = [i for i in range(n)]

        def find(x):
            if root[x] != x:
                root[x] = find(root[x])
            return root[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                root[rx] = ry

        for i, j in pairs:
            union(i, j)

        swap = collections.defaultdict(list)
        for i in range(n):
            heapq.heappush(swap[find(i)], s[i])

        return ''.join([heapq.heappop(swap[root[i]]) for i in range(n)])

    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        def find(x):
            """find the root of node x
            note: in this problem, our edge is weighted, and we need to record them all
            OR, we record and update the value of node x if its root is regard as 1(base value)

            Args:
                x ([type]): [description]

            Returns:
                int: return the value that x should be multiplied to after updating its root 
            """
            if x not in root:
                root[x] = x
                value[x] = 1
            if root[x] != x:
                root[x] = find(root[x])
            return root[x]

        def union(x, y, v):
            rx, ry = find(x), find(y)
            if rx != ry:
                # merge nodes whose root is rx into ry group
                ratio = value[y] * v / value[x]
                for k in value.keys():
                    if find(k) == rx:
                        value[k] *= ratio
                root[rx] = ry

        value = dict()
        root = dict()

        for (i, j), v in zip(equations, values):
            union(i, j, v)

        ans = []
        for x, y in queries:
            if x not in root or y not in root:
                ans.append(-1.0)
            elif find(x) != find(y):
                ans.append(-1.0)
            else:
                ans.append(value[x] / value[y])
        return ans

    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        flowerbed = [1, 0] + flowerbed + [0, 1]
        amount = 0
        for i in flowerbed:
            if i == 1:
                if amount > 0:
                    n -= (amount - 1) // 2
                    amount = 0
                if n <= 0:
                    return True
            else:
                amount += 1
                if (amount - 1) // 2 >= n:
                    return True
        return False

    def ways(self, pizza: List[str], k: int) -> int:
        m, n = len(pizza), len(pizza[0])
        MOD = 10 ** 9 + 7

        # convert pizza to 2d DP matrix
        # dp[i + 1][j + 1] represents for the amount of apple in the area from (0, 0) to (i, j)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i, j in product(range(m), range(n)):
            dp[i + 1][j + 1] = dp[i][j + 1] + \
                dp[i + 1][j] - dp[i][j] + (pizza[i][j] == 'A')

        def cnt_apple(sx, sy, ex, ey):
            return dp[ex + 1][ey + 1] - dp[ex + 1][sy] - dp[sx][ey + 1] + dp[sx][sy]

        @functools.cache
        def cut(i, j, p) -> int:
            """cut the remain pizza into p pieces

            Args:
                i ([type]): remain pizza start from row i
                j ([type]): remain pizza start from col j
                k ([type]): cut into p pieces

            Returns:
                int: return the value of ways to cut
            """
            # remain pizza: (i, j) to (m - 1, n - 1)
            # first, check if the remain pizza has apple on it
            if cnt_apple(i, j, m - 1, n - 1) == 0:
                return 0
            if p == 1:
                return 1

            ans = 0
            # cut horizontally
            # cut between row - 1 and row
            ans += sum(cut(row, j, p - 1) for row in range(i + 1, m)
                       if cnt_apple(i, j, row - 1, n - 1) > 0) % MOD
            # cut vertically
            # cut between col - 1 and col
            ans += sum(cut(i, col, p - 1) for col in range(j + 1, n)
                       if cnt_apple(i, j, m - 1, col - 1) > 0) % MOD
            return ans % MOD

        return cut(0, 0, k)

    def detectCycle(self, head: ListNode) -> ListNode:
        # regardless of O(1) memory
        # visited = set() # key = listnode, value = idx
        # while head:
        #     if head in visited: return head
        #     visited.add(head)
        #     head = head.next
        # return None

        # O(1) space complexity
        if not head:
            return None
        slow = fast = head
        first = True
        while first or slow != fast:
            first = False
            if (not fast.next) or (not fast.next.next):
                return None
            slow = slow.next
            fast = fast.next.next

        while slow != head:
            slow = slow.next
            head = head.next

        return head

    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)
        move = [i - j for i, j in zip(gas, cost)]
        # start = 0 end = n
        # start = -1 end = n - 1
        curr = start = 0
        end = start + 1
        while end - start <= n:
            curr += move[end - 1]
            while curr < 0:
                if start + n < end:
                    return -1
                start -= 1
                curr += move[start]
            end += 1
        return start + n if start < 0 else start

    def getDecimalValue(self, head: ListNode) -> int:
        """convert binary representation from a linkedlist into decimal integer

        Args:
            head (ListNode): a linkedlist that represents for a binary number, from MSB to LSB

        Returns:
            int: the decimal number
        """
        ans = 0
        while head:
            ans = ans * 2 + head.val
            head = head.next
        return ans

    def findRelativeRanks(self, score: List[int]) -> List[str]:
        """convert elements from a list into their ranks

        Args:
            score (List[int]): input list consists of scores of athletes

        Returns:
            List[str]: the ranks number of all athletes, and give medals to the first three ones 
        """
        for rank, (_, index) in enumerate(sorted([[s, idx] for idx, s in enumerate(score)], key=lambda x: -x[0])):
            if rank == 0:
                score[index] = 'Gold Medal'
            elif rank == 1: 
                score[index] = 'Silver Medal'
            elif rank == 2:
                score[index] = 'Bronze Medal'
            else:
                score[index] = str(rank + 1)
        return score
    
    def minDepth(self, root: TreeNode) -> int:
        """find the minimum depth of a binary tree

        Args:
            root (TreeNode): the root node of this tree

        Returns:
            int: the minimum depth of this tree
        """
        if not root: return 0
        curr = [[1, root]]
        ans = float('inf')
        while curr:
            depth, node = curr.pop(0)
            if depth >= ans: continue
            else:
                if not node.left and not node.right:
                    ans = min(ans, depth)
                else:
                    if node.left: curr.append([depth + 1, node.left])
                    if node.right: curr.append([depth + 1, node.right])
        return ans

    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        group = dict()
        dislike = dict()
        for i, j in dislikes:
            if i not in group and j not in group:
                # encounter two new people that has not been seem before
                dislike[i] = j
                dislike[j] = i
                group[i] = i
                group[j] = j
            elif i in group and j not in group:
                group[j] = dislike[group[i]]
            elif i not in group and j in group:
                group[i] = dislike[group[j]]
            else:
                # both can be found be group
                gi, gj = group[i], group[j]
                if gi == gj: return False
                if dislike[gi] == gj: continue
                else:
                    # move (gj, dislike[gj]) to (dislike[gi], gi)
                    for i, g in group.items():
                        if g == gj:
                            group[i] = dislike[gi]
                        elif g == dislike[gj]:
                            group[i] = gi
        return True

    def detectCapitalUse(self, word: str) -> bool:
        return True if word == '' or word.isupper() or word.islower() or word == word.capitalize() else False

    def maximumGood(self, statements: List[List[int]]) -> int:
        ans = 0
        n = len(statements)
        def check(b):
            for i, p in enumerate(b):
                if p == '1': 
                    # person i is good person
                    for idx, s in enumerate(statements[i]):
                        # s == 0 && b[idx] = 0
                        # s == 1 && b[idx] = 1
                        # s == 2
                        if s != 2 and s != int(b[idx]): return False
            return True

        for i in range(2 ** n):
            bitmask = bin(i)[2:].zfill(n)
            if check(bitmask):
                ans = max(ans, bitmask.count('1'))
        return ans

    def maxScore(self, cardPoints: List[int], k: int) -> int:
        """return the maximum points you can get by removing k cards, from both sides

        Args:
            cardPoints (List[int]): represents for the points of each card
            k (int): amount of cards that you need to take

        Returns:
            int: the maximum points you can get
        """
        # minimize the remain points
        k = len(cardPoints) - k
        ans = sum(cardPoints[:k])
        ret = ans
        for i in range(1, len(cardPoints) + 1 - k):
            # sum of cardPoints[i, i + k]
            # + cardPoints[i + k - 1] - cardPoints[i - 1]
            ans += cardPoints[i + k - 1] - cardPoints[i - 1]
            ret = min(ret, ans)
        return sum(cardPoints) - ret

    def validMountainArray(self, arr: List[int]) -> bool:
        if len(arr) < 3 or arr[1] <= arr[0]: return False
        climb = True
        pre = arr[0]
        for i in arr[1:]:
            if i == pre: return False
            if not climb and i > pre: return False
            if climb and i < pre: climb = False
            pre = i
        return not climb

    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        """merge two binary trees

        Args:
            root1 (TreeNode): Tree 1
            root2 (TreeNode): Tree 2

        Returns:
            List[int]: merged Tree in ascending order
        """
        ans = []
        def element(node) -> list:
            if node: 
                ans.append(node.val)
                element(node.left)
                element(node.right)
        
        element(root1)
        element(root2)
        return sorted(ans)

    def duplicateZeros(self, arr: List[int]) -> None:
        """duplicate zeros in list and shift elements to right
        Do not return anything, modify arr in-place instead.
        """
        dq = collections.deque()
        for i, v in enumerate(arr):
            dq.append(v)
            arr[i] = dq.popleft()
            if v == 0:
                dq.append(0)

    def superpalindromesInRange(self, left: str, right: str) -> int:
        """an integer is a super-palindrome if it is a palindrome, and it is also the square of a palindrome.

        Args:
            left (str): lower bound
            right (str): upper bound

        Returns:
            int: number of super-palindromes
        """
        left = math.ceil(math.sqrt(int(left)))
        right = math.floor(math.sqrt(int(right)))
        ans = 0

        def ispalindrome(num):
            return str(num) == str(num)[::-1]

        # all digit can only be 0, 1, 2, except for number 3
        nums = [3]
        
        def buildnumbers(pre):
            if pre < right:
                pre *= 10
                nxt = [pre] if pre else []
                nxt += [pre + 1, pre + 2]
                for nx in nxt:
                    nums.append(nx)
                    buildnumbers(nx)

        buildnumbers(0)
        for i in nums:
            if right >= i >= left and ispalindrome(i) and ispalindrome(i ** 2): 
                print(i, i ** 2)
                ans += 1
        return ans

    def findMaximumXOR(self, nums: List[int]) -> int:
        class Trie:
            """
            this Trie class is design for 32-bit numbers
            insert method: insert number as 32-bit integer
            find method: find a number that has the most amount of same bits as the given target
            """
            def __init__(self):
                self.tree = dict()

            def insert(self, num: int) -> None:
                temp = self.tree
                for c in bin(num)[2:].zfill(32):
                    if c not in temp: temp[c] = dict()
                    temp = temp[c]
            
            def find(self, target: int) -> int:
                ans = 0
                temp = self.tree
                for c in bin(target)[2:].zfill(32):
                    ans <<= 1
                    if c not in temp: c = '01'[c == '0']
                    ans += int(c)
                    temp = temp[c]
                return ans    

        ans = 0
        tr = Trie()
        for n in nums:
            tr.insert(n)
        
        for n in nums:
            match = tr.find(((1 << 32) - 1) ^ n)
            ans = max(ans, n ^ match)
        return ans

    def findTheDifference(self, s: str, t: str) -> str:
        s = collections.Counter(s)
        for i in t:
            if i not in s or s[i] == 0: return i
            s[i] -= 1
    
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        uf = UnionFind(n)
        for i in range(n):
            for j in range(i + 1, n):
                if isConnected[i][j] == 1:
                    uf.union(i, j)
        
        return uf.getCount()
    
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        uf = UnionFind(n)
        for i, j in pairs:
            uf.union(i, j)
        
        s = list(s)
        idx = collections.defaultdict(list)
        char = collections.defaultdict(list)
        for i in range(n):
            root = uf.find(i)
            idx[root].append(i)
            char[root].append(s[i])
        
        for k in idx.keys():
            for i, c in zip(sorted(idx[k]), sorted(char[k])):
                s[i] = c
        
        return ''.join(s)
    
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        root = dict()
        value = dict()

        # The find function here is the same as that in the disjoint set with path compression.
        def find(x):
            if x != root[x]:
                root[x] = find(root[x])
            return root[x]

        def union(x, y, k):
            if x not in root:
                root[x] = x
                value[x] = 1
            if y not in root:
                root[y] = y
                value[y] = 1
            rootX, rootY = find(x), find(y)
            if rootX != rootY:
                diff = value[y] * k / value[x]
                for key in root.keys():
                    if find(key) == rootX:
                        value[key] *= diff
                root[rootX] = rootY
            
        def calc(x, y):
            return -1 if x not in root or y not in root or find(x) != find(y) else value[x] / value[y]
        
        for [i, j], k in zip(equations, values):
            union(i, j, k)
        
        return [calc(i, j) for i, j in queries]

    def addDigits(self, num: int) -> int:
        """repeatedly add all its digits until the result has only one digit

        Args:
            num (int): given number in range(2 ** 31)

        Returns:
            int: final number which only has one digit
        """

        # to solve it in O(1) time
        # we have to find out the rule of result
        # for i in range(30):
        #     while i > 9:
        #         i = eval('+'.join(list(str(i))))
        #     print(i, end='')
        # Output:
        # 012345678912345678912345678912
        return 0 if not num else [num % 9, 9][num % 9 == 0]

    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        """find if there is a path from source vertex to the destination vertex

        Args:
            n (int): Ummm, useless
            edges (List[List[int]]): [description]
            source (int): [description]
            destination (int): [description]

        Returns:
            bool: [description]
        """

        # DFS & BFS
        graph = collections.defaultdict(list)

        for i, j in edges:
            graph[i].append(j)
            graph[j].append(i)
        
        visited = set()
        curr = collections.deque([source])

        while curr:
            # popleft() to implement a BFS solution
            # while pop() to implement a DFS solution
            temp = curr.popleft()
            if temp == destination: return True
            for i in graph[temp]:
                if i not in visited:
                    visited.add(i)
                    curr.append(i)
        return False

        # disjoint set
        uf = UnionFind(n)
        for i, j in edges:
            uf.union(i, j)
        return uf.connected(source, destination)
    
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        """find all paths from source to target

        Args:
            graph (List[List[int]]): a list of edges in a DAG

        Returns:
            List[List[int]]: all paths from 0 to n - 1
        """
        path = collections.defaultdict(list)
        n = len(graph)
        for i in range(n):
            for j in graph[i]:
                path[i].append(j)

        ans = []
        def move(pre, pre_set):
            if pre[-1] == n - 1:
                ans.append(pre)
                return
            for nxt in path[pre[-1]]:
                if nxt not in pre_set:
                    move(pre + [nxt], pre_set | {nxt})
            
        move([0], {0})
        return ans
    
    def cloneGraph(self, node: Node) -> Node:
        """
        # Definition for a Node.
        class Node:
            def __init__(self, val = 0, neighbors = None):
                self.val = val
                self.neighbors = neighbors if neighbors is not None else []
        """
        if not node: return None
        cp = {node: Node(node.val, [])}
        curr = collections.deque([node])

        while curr:
            temp = curr.popleft()
            for ne in temp.neighbors:
                if ne not in cp:
                    ne_cp = Node(ne.val, [])
                    cp[ne] = ne_cp
                    curr.append(ne)
                cp[temp].neighbors.append(cp[ne])
        return cp[node]

    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        """
        find the smallest lexical order itinerary for a man who departs from "JFK"
        all tickets should be used once and only once

        Args:
            tickets (List[List[str]]): list of tickets

        Returns:
            List[str]: fly path
        """
        path = collections.defaultdict(dict) # key: departure airport | value: dict -> {arrival airport: cnt}
        for i, j in tickets:
            if j not in path[i]:
                path[i][j] = 0
            path[i][j] += 1

        def fly(curr, cnt):
            if cnt == 0: return [curr]
            for k in sorted(path[curr].keys()):
                if path[curr][k] == 0:
                    continue
                path[curr][k] -= 1
                ret = fly(k, cnt - 1)
                if ret: return [curr] + ret
                path[curr][k] += 1
            return None
        
        return fly('JFK', len(tickets))

    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        """find the shortest path from top-left cell to bottom-right cell

        Args:
            grid (List[List[int]]): map 1 = block & 0 = empty

        Returns:
            int: length of shortest path
        """
        if grid[0][0] == 1: return -1
        dir = [[0, 1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [1, -1], [1, 0], [1, 1]]
        pq = [(1, 0, 0)]
        m, n = len(grid), len(grid[0])
        while pq:
            cnt, x, y = heapq.heappop(pq)
            if x == m - 1 and y == n - 1: return cnt
            for i, j in dir:
                xi, yj = x + i, y + j
                if m > xi >= 0 <= yj < n and grid[xi][yj] == 0:
                    grid[xi][yj] = 1
                    heapq.heappush(pq, (cnt + 1, xi, yj))
        return -1
    
    def levelOrder(self, root: Node) -> List[List[int]]:
        """
        # Definition for a Node.
        class Node:
            def __init__(self, val=None, children=None):
                self.val = val
                self.children = children
        """
        ans = list()
        if root: 
            curr = [root]
            while curr:
                ans.append([])
                temp = list()
                for i in curr:
                    ans[-1].append(i.val)
                    if i.children: temp += i.children
                curr = temp
        return ans

    def gridIllumination(self, n: int, lamps: List[List[int]], queries: List[List[int]]) -> List[int]:
        light_row = collections.defaultdict(int)
        light_col = collections.defaultdict(int)
        light_diag = collections.defaultdict(int)
        light_rever = collections.defaultdict(int)

        dir = [[0, 1], [0, -1], [-1, -1], [-1, 1], [1, 0], [-1, 0], [1, -1], [1, 1]]
        lamps = {(i, j) for i, j in lamps}

        # since board size is up to 10 ** 9, if we update cell one by one
        # this turn method will take O(10 ** 9) time, which will result in TLE
        # optimize time complexity to O(1)
        # do not update single cell
        # instead of that, we update row idx, col idx, diagonal indices
        def turn(x, y, act):
            light_row[x] += act
            light_col[y] += act
            light_diag[y - x + n] += act
            light_rever[x + y] += act

        for i, j in lamps:
            turn(i, j, 1)

        ans = list()
        for i, j in queries:
            ans.append([0, 1][light_row[i] + light_col[j] + light_diag[j - i + n] + light_rever[i + j] > 0])
            for x, y in dir + [[0, 0]]:
                xi, yj = x + i, y + j
                if (xi, yj) in lamps:
                    lamps.remove((xi, yj))
                    turn(xi, yj, -1)
        return ans

    def findPairs(self, nums: List[int], k: int) -> int:
        if k == 0: return sum(1 for i in collections.Counter(nums).values() if i > 1)
        nums = set(nums)
        visited = set()
        ans = 0
        for i in nums:
            if i - k in visited: ans += 1
            if i + k in visited: ans += 1
            visited.add(i)

        return ans

    def countKDifference(self, nums: List[int], k: int) -> int:
        nums = collections.Counter(nums)
        visited = collections.defaultdict(int)

        ans = 0
        for i, cnt in nums.items():
            if i + k in visited: ans += cnt * visited[i + k]
            if i - k in visited: ans += cnt * visited[i - k]
            visited[i] = cnt
        return ans

    def checkInclusion(self, s1: str, s2: str) -> bool:
        """check if one of s1's permutation is the substring of s2

        Args:
            s1 (str): given string 1
            s2 (str): given string 2

        Returns:
            bool: return True if we found the permutation
        """
        cnt = collections.Counter(s1)
        start = end = 0
        print(cnt)
        while end < len(s2):
            if s2[end] not in cnt:
                start = end = end + 1
                cnt = collections.Counter(s1)
                continue
            cnt[s2[end]] -= 1
            while cnt[s2[end]] < 0:
                cnt[s2[start]] += 1
                start += 1
            if end - start + 1 == len(s1): return True
            end += 1 
        return False

    def minimumDifference(self, nums: List[int], k: int) -> int:
        """pick k numbers from nums

        Args:
            nums (List[int]): given numbers
            k (int): pick k elements from nums

        Returns:
            int: return the minimum possible diff between the smallest and the greatest number 
        """
        if k == 1: return 0
        ans = float('inf')
        nums.sort()
        for i, j in zip(nums[k - 1:], nums):
            ans = min(ans, i - j)
            if ans == 0: return ans
        return 
        
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        """minimum spanning tree

        Args:
            points (List[List[int]]): a list of all points represents in 2d point format

        Returns:
            int: cost of the minimum spanning tree
        """
        # Prim's algorithm
        n = len(points)
        graph = collections.defaultdict(dict) # key: vertex | velue: [connected vertex, cost]
        # start with an arbitrary vertex, let's choose 0
        distance = [(0, 0)]
        visited = set()

        for i in range(n):
            for j in range(i + 1, n):
                dis = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
                graph[i][j] = dis
                graph[j][i] = dis
        
        ans = 0
        # select vertex with the least cost from unselected vertices
        for _ in range(n):
            while distance[0][1] in visited:
                heapq.heappop(distance)
            cost, vertex = heapq.heappop(distance)
            visited.add(vertex)
            ans += cost
            for nxt, dis in graph[vertex].items():
                heapq.heappush(distance, (dis, nxt))
        return ans

        # Kruskal's algorithm
        n = len(points)
        edge = [] # heapq, element = [distance, x, y]
        uf = UnionFind(n)

        for i in range(n):
             for j in range(i + 1, n):
                heapq.heappush(edge, [abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1]), i, j])
        
        ans = 0
        for _ in range(n - 1): # we need n - 1 edges to connect n nodes
            while uf.connected(edge[0][1], edge[0][2]):
                heapq.heappop(edge)
            cost, x, y = heapq.heappop(edge)
            uf.union(x, y)
            ans += cost
        return ans

    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        edge = collections.defaultdict(dict)
        for s, e, t in times:
            edge[s][e] = t
        shortest = dict() # key: vertex | value: time
        for i in range(1, n + 1):
            shortest[i] = float('inf')
        shortest[k] = 0

        ans = 0
        while shortest:
            v, k = sorted([v, k] for k, v in shortest.items())[0]
            shortest.pop(k)
            ans = max(ans, v)
            for neighbor, cost in edge[k].items():
                if neighbor in shortest:
                    shortest[neighbor] = min(shortest[neighbor], v + cost)
        return ans if ans < float('inf') else -1

    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        source = collections.defaultdict(dict)
        for s, d, cost in flights:
            source[d][s] = cost
        
        dp = [float('inf')] * n
        dp[src] = 0

        for _ in range(1, k + 2):
            nxt = dp.copy()
            for i in range(n):
                for s, c in source[i].items():
                    nxt[i] = min(nxt[i], dp[s] + c)
            dp = nxt
        return dp[dst] if dp[dst] < float('inf') else -1
    
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        m, n = len(heights), len(heights[0])
        def find(k):
            # check if we can find a path with edges at most cost k
            curr = collections.deque([[0, 0]])
            visited = [[False] * n for _ in range(m)]
            visited[0][0] = True
            while curr:
                x, y = curr.popleft()
                if x == m - 1 and y == n - 1: return True
                for i, j in dir:
                    xi, yj = x + i, y + j
                    if m > xi >= 0 <= yj < n and not visited[xi][yj] and abs(heights[xi][yj] - heights[x][y]) <= k:
                        visited[xi][yj] = True
                        curr.append([xi, yj])
            return False

        left, right = 0, max(max(i) for i in heights) - min(min(i) for i in heights)
        while left < right:
            mid = (right + left) >> 1
            if find(mid):
                right = mid
            else:
                left = mid + 1
        return left

    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        in_degree = [0] * numCourses
        visited = set()
        nxt = collections.defaultdict(list)
        for c, p in prerequisites:
            nxt[p].append(c)
            in_degree[c] += 1
        
        ans = list()
        while len(visited) < numCourses:
            update = list()
            for idx, i in enumerate(in_degree):
                if i == 0 and idx not in visited:
                    update.append(idx)
                    visited.add(idx)
                    for n in nxt[idx]:
                        in_degree[n] -= 1
            if not update: return []
            ans += update
        return ans
        
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        connect = collections.defaultdict(list)

        for s, t in edges:
            connect[s].append(t)
            connect[t].append(s)

        leaves = set()
        for i in range(n):
            if len(connect[i]) <= 1:
                leaves.add(i)
        
        left = n
        while left > 2:
            left -= len(leaves)
            nxt = set()
            while leaves:
                temp = leaves.pop()
                neighbor = connect[temp].pop()
                connect[neighbor].remove(temp)
                if len(connect[neighbor]) == 1:
                    nxt.add(neighbor)
                    
            leaves = nxt
        return leaves

    def lastStoneWeight(self, stones: List[int]) -> int:
        pq = [-s for s in stones]
        heapq.heapify(pq)
        while len(pq) > 1:
            y, x = -heapq.heappop(pq), -heapq.heappop(pq)
            if x == y:
                continue
            else:
                heapq.heappush(pq, -(y - x))
        
        return 0 if not pq else -pq[0]

    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        # anti-weakest heap
        # element = (-cnt, -index)
        pq = []
        for idx, row in enumerate(mat):
            cnt = sum(row)
            if len(pq) < k:
                heapq.heappush(pq, (-cnt, -idx))
            else:
                heapq.heappushpop(pq, (-cnt, -idx))
        ans = []
        while pq:
            ans.append(-heapq.heappop(pq)[1])
        return ans[::-1]
            
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        # heapq: element = (number, row index, col index)
        m, n = len(matrix), len(matrix[0])
        pq = [(matrix[i][0], i, 0) for i in range(m)]
        heapq.heapify(pq)

        for _ in range(k):
            i, row, col = heapq.heappop(pq)
            if col + 1 < n:
                heapq.heappush(pq, (matrix[row][col + 1], row, col + 1))
        return i 

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        # anti-closest heapq: element = (distance, x, y)
        pq = []
        for x, y in points:
            if len(pq) < k:
                heapq.heappush(pq, (-(x ** 2 + y ** 2) ** 0.5, x, y))
            else:
                heapq.heappushpop(pq, (-(x ** 2 + y ** 2) ** 0.5, x, y))
        return [[x, y] for _, x, y in pq]

    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        # spend ladders for 'ladders' highest jumps
        # sum up the rest jumps, if the sum is less than or equal to bricks, then we can reach here
        jumps = []
        total_bri = 0
        n = len(heights)
        for i in range(n - 1):
            diff = heights[i + 1] - heights[i]
            if diff <= 0: continue
            if len(jumps) < ladders:
                heapq.heappush(jumps, diff)
            else:
                total_bri += heapq.heappushpop(jumps, diff)
                if total_bri > bricks: return i
        return n - 1
    
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # dp
        dp = [0, 0] # pre_2, pre_1
        co = cost[:2]
        for i in range(2, len(cost)):
            dp = [dp[1], min(pre + c for pre, c in zip(dp, co))]
            co = [co[1], cost[i]]
        return min(pre + c for pre, c in zip(dp, co))

    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        # bottom-up dp
        # dp[i][j] represents for the max score we can get for input nums[start:end]
        n = len(nums)
        m = len(multipliers)
        dp = [[-float('inf')] * (n + 1) for _ in range(n + 1)]
        # return dp[0][n]
        multipliers = [0] * (n - m) + multipliers[::-1]

        for i in range(n + 1): dp[i][i] = 0
        
        for diff in range(1, n + 1):
            for i in range(n):
                if i + diff > n: break
                if diff <= n - m: dp[i][i + diff] = 0
                else:
                    dp[i][i + diff] = max(dp[i + 1][i + diff] + nums[i] * multipliers[diff - 1], dp[i][i + diff - 1] + nums[i + diff - 1] * multipliers[diff - 1])
        
        return dp[0][n]

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = 0
        for i in range(n + 1):
            dp[0][i] = 0

        for i, j in product(range(m), range(n)):
            if text1[i] == text2[j]:
                dp[i + 1][j + 1] = 1 + dp[i][j]
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
        
        return dp[-1][-1]
    
    def maxDepth(self, root: TreeNode) -> int:
        def depth(node):
            if not node: return 0
            else:
                return 1 + max(depth(node.left), depth(node.right))
        
        return depth(root)

    def countOperations(self, num1: int, num2: int) -> int:
        ans = 0
        if num1 > num2: num1, num2 = num2, num1
        while num1 > 0:
            ans += num2 // num1
            num1, num2 = num2 % num1, num1
        return ans

    def minimumOperationsII(self, nums: List[int]) -> int:
        if len(nums) == 1: return 0
        # all odd-indexed elements and even ones should be the same
        odd = collections.Counter([i for idx, i in enumerate(nums) if idx % 2])
        even = collections.Counter([i for idx, i in enumerate(nums) if idx % 2 == 0])
        odd[float('inf')] = 0
        even[float('inf')] = 0
        odd_pq = [[-cnt, value] for value, cnt in odd.items()]
        even_pq = [[-cnt, value] for value, cnt in even.items()]
        heapq.heapify(odd_pq)
        heapq.heapify(even_pq)
        if odd_pq[0][1] != even_pq[0][1]: return len(nums) + odd_pq[0][0] + even_pq[0][0]
        o1, _ = heapq.heappop(odd_pq)
        o2, _ = heapq.heappop(odd_pq)
        e1, _ = heapq.heappop(even_pq)
        e2, _ = heapq.heappop(even_pq)
        return len(nums) + min(o1 + e2, o2 + e1)
    
    def singleNumber(self, nums: List[int]) -> int:
        visited = set()
        for i in nums:
            if i in visited: visited.remove(i)
            else: visited.add(i)
        return visited[0]
    
    def swapPairs(self, head: ListNode) -> ListNode:
        ans = ListNode()
        ans.next = head
        
        pre, curr = ans, head
        while curr and curr.next:
            nxt, remain = curr.next, curr.next.next

            # swap middle two nodes
            pre.next = nxt
            nxt.next = curr
            curr.next = remain

            # reassign all pointers
            pre = curr
            curr = remain
        return ans.next

    def containsDuplicate(self, nums: List[int]) -> bool:
        visited = set()
        for i in nums:
            if i in visited: return True
            visited.add(i)
        return False

    def maxSubArray(self, nums: List[int]) -> int:
        pre_min = 0
        ans = nums[0]
        s = 0
        for i in nums:
            s += i
            ans = max(ans, s - pre_min)
            pre_min = min(pre_min, s)
        return ans

    def countPairs(self, deliciousness: List[int]) -> int:
        MOD = 10 ** 9 + 7
        cnt = collections.Counter(deliciousness)
        power = [2 ** i for i in range(22)]
        ans = 0
        for k, v in cnt.items():
            for p in power:
                if p - k in cnt:
                    if k == p - k:
                        ans += cnt[k] * (cnt[k] - 1) // 2
                    else:
                        ans += cnt[k] * cnt[p - k]
                    ans %= MOD
            cnt[k] = 0
        return ans

    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        boxTypes = [(-j, i) for i, j in boxTypes]
        heapq.heapify(boxTypes)
        ans = 0
        while truckSize and boxTypes:
            unit, cnt = heapq.heappop(boxTypes)
            cnt = min(cnt, truckSize)
            ans += -unit * cnt
            truckSize -= cnt
        return ans

    def waysToSplit(self, nums: List[int]) -> int:
        """
        split nums into three non-empty contiguous subarrays - named left, mid, right
        sum(left) < sum(mid) < sum(right)
        nums:     [--left--] [---------mid----------] [-------right-------]
        subarrays:      left left + 1   (low <- right right + 1 -> high)

        time complexity = O(NlogN)
        space complexity = O(1) extra space

        Args:
            nums (List[int]): input array

        Returns:
            int: number of split ways 
        """
        MOD = 10 ** 9 + 7

        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        
        # find the lower boundary for left = idx
        # binary search: return the smallest index to ensure that sum(mid) >= sum(left)
        def find_low(idx):
            start, end = idx + 1, len(nums) - 1
            while start < end:
                mid = (start + end) >> 1
                if nums[mid] - 2 * nums[idx] >= 0:
                    end = mid
                else:
                    start = mid + 1
            return start

        # find the higher boundary for left = idx
        # binary searh: return the biggest index to ensure that sum(right) >= sum(mid)
        def find_high(idx):
            start, end = idx + 1, len(nums) - 1
            while start < end:
                mid = (start + end) >> 1
                if nums[-1] - nums[mid] >= nums[mid] - nums[idx]:
                    start = mid + 1
                else:
                    end = mid
            return start

        # [:left + 1] [left + 1:right + 1] [right + 1:]
        ans = 0
        for left in range(len(nums)):
            if nums[-1] - nums[left] < 2 * nums[left]: break
            ans = (ans + max(0, find_high(left) - find_low(left))) % MOD
        return ans

    def minOperations(self, target: List[int], arr: List[int]) -> int:
        h = {a: i for i, a in enumerate(target)}
        stack = []
        for a in arr:
            if a not in h: continue
            i = bisect.bisect_left(stack, h[a])
            if i == len(stack):
                stack.append(h[a])
            else:
                stack[i] = h[a]
        return len(target) - len(stack)

        # TLE
        # longest common subsequence
        # len(lcs)
        # return len(target) - len(lcs)
        m, n = len(target), len(arr)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i, j in product(range(m), range(n)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if target[i] == arr[j] else max(dp[i][j + 1], dp[i + 1][j])
        return m - dp[-1][-1]
        
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # dfs
        ans = []
        n = len(candidates)
        def extend(idx, pre, remain):
            if remain == 0: 
                ans.append(pre)
                return
            if remain < 0 or idx == n: return
            extend(idx, pre + [candidates[idx]], remain - candidates[idx])
            extend(idx + 1, pre, remain)

        extend(0, [], target)
        return ans

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        visited = dict()
        for i, v in enumerate(nums):
            if target - v in visited:
                return [visited[target - v], i]
            visited[v] = i

    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        stack = collections.deque([])

        i = j = 0
        while i < m or j < n:
            if j == n or (i < m and nums1[i] <= nums2[j]):
                stack.append(nums1[i])
                nums1[i] = stack.popleft()
                i += 1
            elif i == m or (j < n and nums1[i] > nums2[j]):
                stack.append(nums2[j])
                j += 1
        while i < m + n:
            nums1[i] = stack.popleft()
            i += 1
        
    def canMeasureWater(self, jug1Capacity: int, jug2Capacity: int, targetCapacity: int) -> bool:
        if targetCapacity > jug1Capacity + jug2Capacity: return False

        def gcd(x, y):
            if x > y: x, y = y, x
            while x > 0:
                x, y = y % x, x
            return y

        return targetCapacity % gcd(jug1Capacity, jug2Capacity) == 0

    