#!/usr/bin/env python3
from typing import List


class Assessment:
    def restoreString(self, s: str, indices: List[int]) -> str:
        return "".join([ch for _, ch in sorted(zip(indices, s))])

    def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
        return sum([max(abs(i[0] - pre[0]), abs(i[1] - pre[1])) for pre, i in zip(points, points[1:])])
