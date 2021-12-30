#!/usr/bin/env python3
class CombinationIterator:

    def __init__(self, characters: str, combinationLength: int):
        # since the input are sorted distinct, just iterate normally
        self.comb = list()

        def find_next(pre: str, idx: int):
            if len(pre) == combinationLength:
                self.comb.append(pre)
                return
            if idx == len(characters): return
            for i in range(idx, len(characters)):
                find_next(pre + characters[i], i + 1)
        
        find_next('', 0)

    def next(self) -> str:
        return self.comb.pop(0)

    def hasNext(self) -> bool:
        return True if self.comb else False


# Your CombinationIterator object will be instantiated and called as such:
# obj = CombinationIterator(characters, combinationLength)
# param_1 = obj.next()
# param_2 = obj.hasNext()