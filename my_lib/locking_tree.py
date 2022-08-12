#!/usr/bin/env python3
import collections
from typing import List


class LockingTree:
    def __init__(self, parent: List[int]):
        n = len(parent)
        # node_state[i] = 1 if node i is locked
        self.node_state = [0] * n
        # last_user[i] = 2 if user 2 is the latest user access node i
        self.last_user = [0] * n
        # all children of node i
        self.children = collections.defaultdict(list)
        # all ancestors of node i
        self.ancestors = collections.defaultdict(list)

        # group parent in a dict, where key is parent, value is a list of all nodes share this parent
        direct_children = collections.defaultdict(list)
        for id, pa in enumerate(parent):
            direct_children[pa].append(id)
        # iterate parent in following way:
        # 0(root) first, put 0 to visited set, and put 0's direct children in next_iter list
        # iterate next_iter list...
        visited = set()
        next_iter = [0]
        while len(visited) < len(parent):
            n_next = len(next_iter)
            for i in range(n_next):
                temp = next_iter.pop(0)
                visited.add(temp)
                if temp > 0:
                    # append parent to ancestors of this parent
                    self.ancestors[temp] = [parent[temp]] + \
                        self.ancestors[parent[temp]]
                    for ances in self.ancestors[temp]:
                        self.children[ances].append(temp)
                next_iter += direct_children[temp]
        print(self.ancestors)
        print(self.children)
        super().__init__()

    def lock(self, num: int, user: int) -> bool:
        # node is unlocked and all its ancestors are unlocked
        if self.node_state[num] == 0:
            self.node_state[num] = 1
            self.last_user[num] = user
            return True
        return False

    def unlock(self, num: int, user: int) -> bool:
        # node is currently locked by the same user
        if self.node_state[num] == 1 and self.last_user[num] == user:
            self.node_state[num] = 0
            return True
        return False

    def upgrade(self, num: int, user: int) -> bool:
        # The node is unlocked,
        # It has at least one locked descendant (by any user), and
        # It does not have any locked ancestors.
        if self.node_state[num] == 0 and any(self.node_state[child] == 1 for child in self.children[num]) and all(self.node_state[ances] == 0 for ances in self.ancestors[num]):
            self.node_state[num] = 1
            self.last_user[num] = user
            for child in self.children[num]:
                self.node_state[child] = 0
            return True
        return False


if __name__ == '__main__':
    obj = LockingTree([-1, 4, 1, 2, 8, 0, 8, 0, 0, 7])
    print(obj.lock(2, 2))
    print(obj.unlock(2, 3))
    print(obj.unlock(2, 2))
    print(obj.lock(4, 5))
    print(obj.upgrade(0, 1))
    print(obj.lock(0, 1))
# ["LockingTree","upgrade","upgrade","unlock","lock","upgrade"]
# [[[-1,0,3,1,0]],[4,5],[3,8],[0,7],[2,7],[4,6]]
# ["LockingTree","lock","unlock","unlock","lock","upgrade","lock"]
# [[[-1,0,0,1,1,2,2]],[2,2],[2,3],[2,2],[4,5],[0,1],[0,1]]
# ["LockingTree","lock","unlock","upgrade","upgrade","unlock","upgrade","upgrade","upgrade","lock","upgrade","upgrade","unlock","upgrade","unlock","unlock","unlock","upgrade","lock","lock","lock"]
# [[[-1,4,1,2,8,0,8,0,0,7]],[8,48],[8,48],[4,47],[8,16],[4,23],[7,39],[6,39],[9,33],[5,32],[8,8],[6,5],[6,42],[5,19],[3,45],[7,45],[1,25],[0,15],[5,42],[5,16],[4,25]]

# Your LockingTree object will be instantiated and called as such:
# obj = LockingTree(parent)
# param_1 = obj.lock(num,user)
# param_2 = obj.unlock(num,user)
# param_3 = obj.upgrade(num,user)
