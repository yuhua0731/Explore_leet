#!/usr/bin/env python3
from typing import List
import math

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


