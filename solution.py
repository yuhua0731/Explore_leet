#!/usr/bin/env python3
class Solution:
    def __init__(self) -> None:
        super().__init__()


    def isIsomorphic(self, s: str, t: str) -> bool:
        def findOccurance(ss: str):
            res = dict()
            for i in range(len(ss)):
                if ss[i] not in res:
                    res[ss[i]] = list()
                res[ss[i]].append(i)
            return res

        if len(s) != len(t):
            return False
        dicts, dictt = findOccurance(s), findOccurance(t)
        for value in dicts.values():
            valuet = dictt[t[value[0]]]
            if value != valuet:
                return False
        return True