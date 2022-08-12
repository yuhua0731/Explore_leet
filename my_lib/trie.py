#!/usr/bin/env python3
class Trie:
    # simple 2 dicts without any tree-like structure
    # start = set()
    # words = set()

    # def __init__(self):
    #     self.start.clear()
    #     self.words.clear()

    # def insert(self, word: str) -> None:
    #     for i in range(len(word)):
    #         self.start.add(word[:i + 1])
    #     self.words.add(word)

    # def search(self, word: str) -> bool:
    #     return word in self.words

    # def startsWith(self, prefix: str) -> bool:
    #     return prefix in self.start

    def __init__(self):
        self.tree = dict()

    def insert(self, word: str) -> None:
        temp = self.tree
        for c in word:
            if c not in temp: temp[c] = dict()
            temp = temp[c]
        temp['%'] = word

    def search(self, word: str) -> bool:
        temp = self.tree
        for c in word:
            if c not in temp: return False
            temp = temp[c]
        return '%' in temp

    def startsWith(self, prefix: str) -> bool:
        temp = self.tree
        for c in prefix:
            if c not in temp: return False
            temp = temp[c]
        return True

if __name__ == '__main__':
# Your Trie object will be instantiated and called as such:
    obj = Trie()
    obj.insert("word")
    print(obj.search("word"))
    print(obj.startsWith("wo"))
    print(obj.search("wor"))