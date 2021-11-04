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
    print(solu.maxNumberOfBalloons("loonbalxballpoon"))
    print(solu.reverseOnlyLetters("Test1ng-Leet=code-Q!"))
    print(solu.spiralOrder([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    print(solu.addOperators("103", 3))
    print(solu.addOperators("00", 0))
    print(solu.addOperators("3456237490", 911))
    print(solu.tribonacci(25))
    print(solu.canPartitionKSubsets([18,20,39,73,96,99,101,111,114,190,207,295,471,649,700,1037], 4))
    print(solu.calculateMinimumHP([[-2,-3,3],[-5,-10,1],[10,30,-5]]))
    print(solu.rob([2,1,1,2]))
    print(solu.rob_2([2,1,1,2]))
    print(solu.deleteAndEarn([2,2,3,3,3,4]))
    print(solu.findWords([["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]],["oath","pea","eat","rain"]))
    print(solu.findWords([["a", "a"]],["aaa"]))
    print(solu.canJump([3,2,1,0,4]))
    print(solu.jump([2,3,0,1,4]))
    print(solu.jump([1,2,1,1,1,4,4,1,5,2,3,4,1,4,2,5,2,6,4,4,2,2,5,6,2,3,4,5,4,4,2,3,1,4,1,6,2,3,5,3,6,6,1,2,5,3,3,4,6,1,1,5,3,3,4,5,1,4,2,6,6,4,1,4,1,2,1,4,4,2,1,2,2,5,6,5,4,4,3,6,5,2,5,6,1,4,3,4,3,3,1,2,6,5,3,6,1,2,6,4,2,3,3,4,6,3,5,3,2,3,3,1,3,2,4,1,3,5,1,1,5,2,4,2,2,5,3,4,2,1,3,3,1,2,4,5,4,6,2,5,6,4,6,5,2,2,1,4,6,4,2,4,1,6,3,3,6,1,4,5,4,5,1,2,3,6,1,4,3,2,5,1,5,2,5,1,2,3,3,6,6,3,5,2,6,1,6,4,3,4,1,2,5,1,5,6,5,3,1,5,6,3,6,3,5,6,2,2,6,3,4,1,4,1,1,3,4,1,5,6,5,4,2,5,3,6,4,1,2,3,5,6,5,2,3,6,1,3,4,6,3,2,5,5,1,6,6,6,2,3,5,5,4,5,2,1,6,6,2,5,1,3,2,5,1,2,3,4,1,1,5,1,4,1,2,2,6,1,4,3,2,1,6,5,1,6,2,3,5,3,6,6,5,2,1,4,4,5,3,5,5,1,3,2,6,1,6,6,4,6,5,3,3,1,6,2,6,4,2,4,1,2,2,2,2,1,5,4,3,6,3,2,5,5,4,6,4,1,5,2,4,6,2,4,5,5,3,4,6,6,1,6,6,5,3,1,4,6,5,3,5,3,5,2,3,4,6,2,5,6,6,2,5,6,1,1,5,4,5,6,6,5,5,3,3,4,4,5,2,6,5,1,3,2,3,1,3,1,2,3,5,2,5,3,2,2,3,4,4,2,6,5,1,3,4,6,1,6,4,4,2,4,5,2,5,6,6,1,3,1,1,4,6,5,6,4,1,3,1,1,6,2,6,4,5,5,3,5,3,6,6,2,1,3,2,5,5,3,5,3,3,5,3,2,1,2,2,6,1,6,4,2,2,2,6,2,4,2,5,5,2,3,1,1,5,6,6,3,4,6,2,1,2,1,4,2,5,6,5,5,3,2,1,5,1,3,2,2,5,1,6,1,6,5,6,2,6,3,6,5,1,4,6,3,3,6,6,4,1,4,6,3,4,1,4,2,5,5,5,4,2,5,6,6,3,1,5,4,2,3,6,1,6,4,1,5,5,6,4,5,4,4,6,5,2,5,1,4,3,2,6,1,5,2,6,2,6,1,2,3,5,5,4,4,5,4,2,1,4,1,4,6,1,1,2,6,2,3,6,4,4,5,6,6,4,1,6,3,2,4,1,4,5,5,2,6,6,4,2,5,4,6,6,5,2,4,1,1,4,1,1,4,6,1,5,2,4,6,5,1,6,6,6,2,1,6,1,5,5,4,5,2,3,2,2,2,6,4,6,2,4,6,4,5,1,3,2,4,2,6,6,4,3,3,1,1,4,4,5,5,4,1,6,5,1,3,3,6,5,5,3,6,3,5,2,4,3,4,6,5,2,6,6,1,2,3,4,6,1,5,6,4,6,6,1,1,2,4,6,4,1,1,6,6,2,1,1,2,3,6,5,3,1,6,1,3,6,2,4,5,3,2,5,3,5,5,2,1,3,4,4,6,2,4,3,3,1,5,3,3,1,2,5,2,5,2,2,4,2,2,4,6,3,1,4,2,3,4,2,2,6,3,2,6,3,3,5,5,5,2,3,1,6,5,4,5,2,6,5,2,1,2,2,2,2,2,3,2,6,3,1,5,6,1,4,6,5,3,3,5,5,6,5,1,4,3,5,5,3,4,6,4,6,3,2,1,1,6,2,2,5,5,3,1,3,5,6,3,6,2,5,6,2,1,4,4,2,2,6,2,1,5,6,1,1,3,3,5,5,3,2,5,2,1,3,2,4,3,5,2,5,5,4,1,1,3,4,3,1,3,5,5,4,5,5,1,3,5,4,6,5,4,2,1,2,6,6,4,4,5,6,6,6,3,4,3,5,2,5,6,5,2,1,4,5,3,1,6,4,1,5,4,5,2,5,1,4,2,6,3,3,5,1,3,4,3,3,6,6,5,5,5,4,5,3,6,6,6,4,2,4,4,1,2,2,2,3,2,2,5,6,5,6,3,3,1,1,4,1,6,6,5,3,2,6,5,2,1,6,1,4,6,4,1,2,1,2,5,1,1,6,3,2,5,4,5,2,6,5,6,2,2,1,5,5,1,6,2,1,3,4,5,4,3,1,5,6,5,4,1,2,3,4,2,2,6,2,4,3,2,5,3,2,2,5,6,3,3,2,1,4,5,2,3,2,5,3,1,3,6,3,6,4,2,5,3,6,1,6,5,2,1,5,2,1,1,4,3,3,1,1,2,2,1,1,4,1,6,5,5,6,4,6,6,2,2,2,6,1,1,1,1,5,2,2,1,6,5,6,1,3,1,6,4,1,2,1,5,1,1,3,6,4,5,4,2,3,4,1,5,2,2,1,6,2,3,2,3,3,1,1,4,5,5,3,5,3,6,4,5,4,4,4,2,2,1,4,6,0,0,0,0,0]))
    print(solu.maxProduct([2,3,-2,4]))
    print(solu.maxProduct([-2]))
    print(solu.getMaxLen([1,2,3,5,-6,4,0,10]))
    print(solu.maxScoreSightseeingPair([8,1,5,2,6]))
    print(solu.numSquares(2))
    print(solu.numSquares(13))
    print(solu.numSquares(9999))
    print(solu.wordBreak("applepenapple", ["apple","pen"]))
    print(solu.trap([0,1,0,2,1,0,1,3,2,1,2,1]))
    print(solu.trap([4,2,0,3,2,5]))
    print(solu.maxProfitIII([3,3,5,0,0,3,1,4]))
    print(solu.maxProfitIII([1,2,3,4,5]))
    print(solu.nextGreaterElement([4,1,2], [1,3,4,2]))
    print(solu.matrixBlockSum([[1,2,3],[4,5,6],[7,8,9]], 1))
    print(solu.matrixBlockSum([[67,64,78],[99,98,38],[82,46,46],[6,52,55],[55,99,45]], 1))
    print(solu.reverseWords("this is  an   example"))
    print(solu.uniquePathsWithObstacles([[0,0,0],[0,1,0],[0,0,0]]))
    print(solu.frequencySort("raaeaedere"))
    print(solu.longestPalindrome("bapbab"))
    print(solu.longestPalindromeSubseq("bbbab"))
    print(solu.coinChange([1,2,5], 11))
    print(solu.change(6, [2, 4]))
    print(solu.threeSum([-1,0,1,2,-1,-4]))
    print(solu.threeSum([0]))
    print(solu.orangesRotting([[2,1,1],[1,1,0],[0,1,1]]))
    print(solu.orangesRotting([[2,1,1],[1,1,0],[0,0,1]]))
    print(solu.countValidWords(" 62   nvtk0wr4f  8 qt3r! w1ph 1l ,e0d 0n 2v 7c.  n06huu2n9 s9   ui4 nsr!d7olr  q-, vqdo!btpmtmui.bb83lf g .!v9-lg 2fyoykex uy5a 8v whvu8 .y sc5 -0n4 zo pfgju 5u 4 3x,3!wl  fv4   s  aig cf j1 a i  8m5o1  !u n!.1tz87d3 .9    n a3  .xb1p9f  b1i a j8s2 cugf l494cx1! hisceovf3 8d93 sg 4r.f1z9w   4- cb r97jo hln3s h2 o .  8dx08as7l!mcmc isa49afk i1 fk,s e !1 ln rt2vhu 4ks4zq c w  o- 6  5!.n8ten0 6mk 2k2y3e335,yj  h p3 5 -0  5g1c  tr49, ,qp9 -v p  7p4v110926wwr h x wklq u zo 16. !8  u63n0c l3 yckifu 1cgz t.i   lh w xa l,jt   hpi ng-gvtk8 9 j u9qfcd!2  kyu42v dmv.cst6i5fo rxhw4wvp2 1 okc8!  z aribcam0  cp-zp,!e x  agj-gb3 !om3934 k vnuo056h g7 t-6j! 8w8fncebuj-lq    inzqhw v39,  f e 9. 50 , ru3r  mbuab  6  wz dw79.av2xp . gbmy gc s6pi pra4fo9fwq k   j-ppy -3vpf   o k4hy3 -!..5s ,2 k5 j p38dtd   !i   b!fgj,nx qgif "))
    print(solu.nextBeautifulNumber(620883))
    print(solu.countHighestScoreNodes([-1,2,0,2,0]))
    print(solu.minimumTime(5, [[1,5],[2,5],[3,5],[3,4],[4,5]], [1,2,3,4,5]))
    print(solu.solveChess([["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]))
    print(solu.solveChess([["O","O"],["O","O"]]))
    print(solu.minimumOperations([-849,-611,836,-961,-423,611,-117,-794,117,597,-304,-277,909,614,183,784,-447,-692,-351,951,-895,-559,-76,-620,-153,200,-442,-585,996,807,179,455,304,-533,109,372,-863,486,-182,-463,470,-204,-922,97,-321,451,-212,960,-542,-287,56,-748,-159,926,504,-115,-999,-172,-237,-966,-816,569,-20,-156,-90,-332,-532,360,-777,184,-896,-584,-59,775,-908,884,-340,-715,443,563,-853,-502,269,-1000,906,-751,-424,-355,-839,-7,623,317,148,813,177,-441,-723,-14,157,672,-405,181,-695,465,-406,-247,172,-352,91,444,-590,82,-354,-806,-857,-808,-557,694,217,676,956,240,-285,572,289,942,-887,955,930,210,265,115,-576,940,-18,-821,-915,710,-125,881,-296,-708,-689,554,841,-655,759,979,873,-217,827,574,838,713,-578,535,716,825,-742,-175,498,-261,191,580,-240,-591,628,998,-365,-526,865,-734,-286,469,-246,-731,-219,-881,-160,-924,-80,548,-309,-8,587,-49,664,-934,616,-397,879,-743,1000,619,-635,848,970,43,-634,-226,570,943,-797,55,340,-269,396,953,725,-462,-214,391,875,-575,973,242,-238,-311,-325,93,473,752,-800,-30,-191,652,216,-390,-650,-661,-162,-899,497,234,-439,541,-894,305,-52,-804,978,-398,818,-328,-417,429,-740,-643,-72,67,300,383,521,860,-48,-551,309,170,-603,140,195,577,382,-759,-155,-829,-686,-907,65,531,832,-41,997,-897,-220,-835,-803,-884,-295,920,-248,252,-201,-629,-933,-909,-920,-848,-919,989,-200,-396,-969,245,891,754,166,215,-548,958,826,734,-831,-710,553,508,-691,-947,-637,-263,-229,903,720,-480,96,-693,-702,307,-100,164,985,527,-345,187,-353,385,261,-674,-963,-134,-728,-484,-272,-459,509,763,-363,-523,851,112,-193,-939,-210,705,-419,-892,703,984,870,-918,-28,239,83,886,9,-730,646,132,106,24,-485,634,161,-425,401,536,-306,-536,48,-684,-878,595,-186,-906,944,-952,864,589,-107,-258,-596,511,-184,-846,142,-142,480,-544,-815,718,-719,-994,402,-364,-550,-974,-37,-602,-534,270,786,697,-1,258,-746,-319,790,-361,558,603,-465,-711,-133,468,833,-801,647,392,114,-120,-101,-982,856,557,-677,-380,-128,407,-856,484,264,-552,889,744,671,-671,855,803,674,667,-262,859,-111,-875,122,238,196,894,-205,222,-251,-129,326,-555,808,153,-58,-284,488,218,753,198,355,986,927,-275,520,-323,-185,-854,-256,186,-770,-437,739,-979,150,311,-63,428,377,779,-539,-244,395,702,789,384,410,882,287,-148,-951,477,-486,517,729,622,126,-538,447,-872,853,670,608,-537,765,-33,-218,-291,-956,992,-886,190,-547,-69,-882,-95,-647,-653,-819,722,188,-826,945,302,466,339,795,-818,-790,388,-658,-109,939,-901,-773,641,-50,-732,-469,-411,11,-433,-701,-638,463,-199,-276,-601,-350,-288,81,-744,-613,-429,409,36,846,539,77,-386,271,389,698,731,201,-705,840,-233,412,-60,251,809,513,-750,124,-597,941,931,755,-657,-347,-223,921,-694,976,-371,-427,-198,-847,852,452,588,-154,-531,-636,-326,-830,471,-432,772,849,335,-709,-976,-518,13,749,-873,534,-57,361,-973,102,228,-792,-898,627,-717,343,75,-270,990,-945,-122,-932,-807,208,758,-2,-387,110,928,-5,-273,204,582,-21,246,-225,-788,887,901,156,707,621,460,-374,615,-852,585,450,-786,248,523,29,-962,298,-567,88,-173,-169,-47,152,-301,636,-949,-452,680,843,-780,-330,-348,-91,592,-472,-700,-171,715,673,988,-798,349,506,533,325,682,337,-660,425,-449,-929,770,499,-588,-86,-500,-110,783,229,735,-869,167,-22,-923,-333,-511,5,-726,-995,165,-44,-927,-950,-842,842,-183,-221,332,-478,845,-989,-496,-824,-106,-305,-644,769,905,143,119,136,503,806,-605,-367,-453,231,358,487,581,690,-23,632,-468,118,22,-428,-724,-893,-756,-144,130,-617,-890,-608,-435,-239,742,320,543,-136,-168,-572,-369,966,404,290,441,-564,-187,-921,727,936,137,492,780,-455,834,-97,538,-243,-331,403,556,375,70,757,824,796,431,-510,-123,-209,273,626,-56,544,-444,605,113,-885,829,-622,185,105,-541,524,94,872,-791,95,406,-179,92,-230,203,704,515,552],759,552))
    print(solu.possiblyEquals("internationalization", "i18n"))
    print(solu.possiblyEquals("l123e", "44"))
    print(solu.possiblyEquals("a5b", "c5b"))
    print(solu.possiblyEquals("112s", "g841"))
    print(solu.possiblyEquals("ab", "a2"))
    # print(solu.possiblyEquals("6v528u2u87v189u97v357v88u29v3", "v8u2v1v4v29v33u81u899u34v86v1"))
    # print(solu.possiblyEquals("v541v8u458u7v85u35u84v4v16v", "v5v7u818u965v4u89u48u58v418"))
    print(solu.moveZeroes([0, 1, 2, 0, 10, 4, 0]))
    print(solu.reverseWords("hey there we gonna test this function"))

if __name__ == "__main__":
    main()