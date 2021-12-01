Design the `CombinationIterator` class:

- `CombinationIterator(string characters, int combinationLength)` Initializes the object with a string `characters` of **sorted distinct** lowercase English letters and a number `combinationLength` as arguments.
- `next()` Returns the next combination of length `combinationLength` in **lexicographical order**.
- `hasNext()` Returns `true` if and only if there exists a next combination.

 

**Example 1:**

```java
Input
["CombinationIterator", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
[["abc", 2], [], [], [], [], [], []]
Output
[null, "ab", true, "ac", true, "bc", false]

Explanation
CombinationIterator itr = new CombinationIterator("abc", 2);
itr.next();    // return "ab"
itr.hasNext(); // return True
itr.next();    // return "ac"
itr.hasNext(); // return True
itr.next();    // return "bc"
itr.hasNext(); // return False
```

 

**Constraints:**

- `1 <= combinationLength <= characters.length <= 15`
- All the characters of `characters` are **unique**.
- At most `104` calls will be made to `next` and `hasNext`.
- It's guaranteed that all calls of the function `next` are valid.

#### First approach:

Since the input are sorted distinct, just iterate and form substrings normally.

```python
class CombinationIterator:

    def __init__(self, characters: str, combinationLength: int):
        # dfs
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
```

Runtime: 52 ms, faster than 76.70% of Python3 online submissions for Iterator for Combination.

Memory Usage: 16.2 MB, less than 62.62% of Python3 online submissions for Iterator for Combination.