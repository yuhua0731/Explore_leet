#!/usr/bin/env python3
from typing import List
import csv
import time
import random
import collections
import functools

# Definition for singly-linked list.
class ListNode:
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
        with open(self.timestr+'.csv','w',encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile,delimiter=',')
            writer.writerow(['TimeDiff(us)','Cobid','Length','RTR','Data'])

    def take_writer(self):
        with open(self.timestr+'.csv','w',encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile,delimiter=',')
            writer.writerow('hello')


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


    def numDecodings(self, s: str) -> int:
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


    # python pointer
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
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
            if i > 0 and nums[i] == nums[i - 1]: continue
            l, r = i + 1, n - 1
            ls, rs = nums[i] + nums[l] + nums[l + 1], nums[i] + nums[r] + nums[r - 1]
            if ls > target: r = l + 1
            elif rs < target: l = r - 1
            while l < r:
                sums = nums[i] + nums[l] + nums[r]
                if abs(sums - target) < abs(ret - target): ret = sums
                if sums == target: return sums
                elif sums < target: l += 1
                else: r -= 1
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
                if j >= n :
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
        rank = [0] * (m + n) # confused
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