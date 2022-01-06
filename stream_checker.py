#!/usr/bin/env python3
from typing import List


class StreamChecker:
    def __init__(self, words: List[str]):
        self.tree = dict()
        for word in words:
            temp = self.tree
            for c in word:
                if c not in temp:
                    temp[c] = dict()
                temp = temp[c]
            temp['%'] = word
        self.curr = [self.tree]

    def query(self, letter: str) -> bool:
        temp = [self.tree]
        for d in self.curr:
            if letter in d:
                temp.append(d[letter])
        self.curr = temp
        return any('%' in d for d in temp)


if __name__ == '__main__':
    # Your StreamChecker object will be instantiated and called as such:
    obj = StreamChecker(["cd", "f", "fg", "kl"])
    print(obj.query('q'))
    print(obj.query('f'))
    print(obj.query('c'))
    print(obj.query('d'))
    print(obj.query('k'))
