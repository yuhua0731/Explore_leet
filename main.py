#!/usr/bin/env python3
from solution import ListNode, Solution, TreeNode


def main():
    solu = Solution()
    print(solu.isIsomorphic('apple', 'heelo'))
    print(solu.numDecodingsWithStar('1234**01*0'))
    print(solu.lengthOfLIS([1, 3, 6, 7, 9, 4, 10, 5, 6]))
    print(solu.findLength([1, 2, 3, 2, 1], [3, 2, 1, 4, 7]))
    print(solu.findPeakElement([1, 2, 5, 3, 7, 4, 10]))
    print(solu.fourSum([1, 0, -1, 0, -2, 2], 0))
    print(solu.triangleNumber([2, 2, 3, 4]))
    a = ListNode(1, next=ListNode(2, next=ListNode(
        3, next=ListNode(4, next=ListNode(5)))))
    ListNode.printList(solu.reverseKGroup(a, 2))
    p = solu.shuffle([1, 2, 3, 4, 5, 6])
    print(p.shuffle())
    print(p.reset())
    solu.sortedArrayToBST([1, 2, 3, 4, 5, 7, 10]).printNode()
    print(solu.threeSumClosest([0, -1, -1, 2, 3, 6, -4, -2, 3, 5, -1], -3))
    print(solu.twoSum([1, 2, 3, 4, 7], 10))
    print(solu.largestIsland([[1, 1], [1, 1]]))
    print(solu.subsetsWithDup([4, 4, 4, 1, 4]))
    print(list(solu.powerset([1, 2, 3])))
    a = TreeNode(1)
    a.left = TreeNode(2)
    a.right = TreeNode(2)
    print(solu.pathSum(a, 3))
    print(solu.stoneGame([3, 101, 3]))
    print(solu.matrixRankTransform([[7, 8], [8, 7]]))
    print(solu.groupAnagrams(["ddbdddd", "bcb"]))
    print(solu.canReorderDoubled([-2, 2, -4, 4]))
    print(solu.minWindow("ADBEBANC", "ABC"))
    print(solu.numDecodings('0')) # 0
    print(solu.numDecodings('06')) # 0
    print(solu.numDecodings('2101')) # 1
    print(solu.isValidSudoku([["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]))
    print(solu.solveSudoku([["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]))
    print(solu.rectangleArea([[0,0,2,2],[1,1,2,4]]))
    print(solu.findGCD([1, 2, 3, 5, 10]))
    print(solu.findDifferentBinaryString(["001","000","000"]))
    print(solu.minimizeTheDifference([[1, 2, 3], [3, 4, 0]], 5))
    print(solu.complexNumberMultiply('1+1i', '1+1i'))
    print(solu.judgeSquareSum(4))
    print(solu.judgeSquareSum(2 ** 31 - 1))
    print(solu.isValidSerialization("9,3,4,#,#,1,#,#,2,#,6,#,#"))
    print(solu.isValidSerialization("1"))
    print(solu.isValidSerialization("9,#,#,1"))
    print(solu.findLUSlength(['aaa', 'cc', 'aaa']))
    print(solu.minPatches([1, 2, 2], 5))
    print(solu.minPatches([1, 3], 6))
    print(solu.minPatches([1,7,21,31,34,37,40,43,49,87,90,92,93,98,99], 12))
    print(solu.maxCount(3, 3, [[2,2],[3,3],[3,3],[3,3],[1,2],[3,3],[3,3],[3,1],[2,2],[3,3],[3,3],[3,3]]))
    print(solu.minimumDifference([9,4,1,7], 2))
    print(solu.minimumDifference([9], 1))
    print(solu.kthLargestNumber(["2","21","12","1"], 3))
    print(solu.minSessions([1,2,3], 3))
    print(solu.minSessions([3,1,3,1,1], 8))
    print(solu.minSessions([2,3,3,4,4,4,5,6,7,10], 12))
    print(solu.minSessions([1,1,1,1,1,1,1,1,1,1,1,1,1,1], 14))
    print(solu.numberOfUniqueGoodSubsequences("100101"))
    print(solu.findMin([4,5,6,7,0,1,2]))
    print(solu.arrayNesting([0,2,1]))
    print(solu.arrayNesting([5,4,0,3,1,6,2]))
    print(solu.recoverArray(4, [0,0,5,5,4,-1,4,9,9,-1,4,3,4,8,3,8]))
    print(solu.minTimeToType('bza'))
    print(solu.maxMatrixSum([[-10000,-10000,-10000],[-10000,-10000,-10000],[-10000,-10000,-10000]]))
    for i in solu.generateTrees(3):
        print("TreeNode: ")
        i.printNode()
        print("TreeNode end")
    print(solu.countPaths(7, [[0,6,7],[0,1,2],[1,2,3],[1,3,3],[6,3,3],[3,5,1],[6,5,1],[2,5,1],[0,4,5],[4,6,2]]))
    print(solu.numberOfCombinations('312'))
    print(solu.outerTrees([[1,1],[2,2],[2,0],[2,4],[3,3],[4,2]]))
    print(solu.outerTrees([[1,2],[2,2],[4,2]]))
    print(solu.orderlyQueue('bdcbc', 3))
    print(solu.countQuadruplets([1,1,1,3,5]))
    print(solu.countQuadruplets([3,3,6,4,5]))
    print(solu.countQuadruplets([1,2,3,6]))
    print(solu.numberOfWeakCharacters([[5,5],[6,3],[3,6]]))
    print(solu.numberOfWeakCharacters([[1,5],[10,4],[4,3]]))
    print(solu.firstDayBeenInAllRooms([0,0]))
    print(solu.firstDayBeenInAllRooms([0,0,2]))
    print(solu.firstDayBeenInAllRooms([0,1,2,0]))
    print(solu.gcdSort([1, 2, 3, 100, 7]))
    print(solu.findMiddleIndex([2,3,-1,8,4]))
    print(solu.findMiddleIndex([-1,1,-1]))
    print(solu.findFarmland([[1,0,0],[0,1,1],[0,1,1]]))
    print(solu.orderOfLargestPlusSign(5, [[4,2]]))
    print(solu.reachableNodes([[0,1,10],[0,2,1],[1,2,2]], 6, 3))
    print(solu.reachableNodes([[0,1,4],[1,2,6],[0,2,8],[1,3,1]], 10, 4))

if __name__ == "__main__":
    main()
