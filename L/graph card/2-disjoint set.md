# Disjoint set

## Overview of Disjoint Set

------



Given the vertices and edges between them, how could we quickly check whether two vertices are connected? For example, Figure 5 shows the edges between vertices, so how can we efficiently check if 0 is connected to 3, 1 is connected to 5, or 7 is connected to 8? We can do so by using the “==disjoint set==” data structure, also known as the “==union-find==” data structure. Note that others might refer to it as an algorithm. In this Explore Card, the term “disjoint set” refers to a data structure.



![img](https://leetcode.com/explore/learn/card/Figures/Graph_Explore/Disjoint_Set_1.png)



Figure 5. Each graph consists of vertices and edges. The root vertices are in green

The primary use of disjoint sets is to address the connectivity between the components of a network. The “network“ here can be a computer network or a social network. For instance, we can use a disjoint set to determine if two people share a common ancestor.



### Terminologies

------

- **Parent node**: the direct parent node of a vertex. For example, in Figure 5, the parent node of vertex 3 is 1, the parent node of vertex 2 is 0, and the parent node of vertex 9 is 9.
- **Root node**: a node without a parent node; it can be viewed as the parent node of itself. For example, in Figure 5, the root node of vertices 3 and 2 is 0. As for 0, it is its own root node and parent node. Likewise, the root node and parent node of vertex 9 is 9 itself. Sometimes the root node is referred to as the head node.



### Introduction to Disjoint Sets

------

#### Summary of video content:

1. How do “disjoint sets” work.
2. Solving the connectivity question in Figure 5.

video is unaccessible :(

### Implementing “disjoint sets”

------

#### Summary of video content:

1. How to implement a “disjoint set”.
2. The `find` function of a disjoint set.
3. The `union` function of a disjoint set.

video is unaccessible :(

### The two important functions of a “disjoint set.”

------

In the introduction videos above, we discussed the two important functions in a “disjoint set”.

- **The `find` function** finds the root node of a given vertex. For example, in Figure 5, the output of the find function for vertex 3 is 0.
- **The `union` function** unions two vertices and makes their root nodes the same. In Figure 5, if we union vertex 4 and vertex 5, their root node will become the same, which means the union function will modify the root node of vertex 4 or vertex 5 to the same root node.



### There are two ways to implement a “disjoint set”.

------

- Implementation with Quick Find: in this case, the time complexity of the `find` function will be `O(1)`. However, the `union` function will take more time with the time complexity of `O(N)`.
- Implementation with Quick Union: compared with the Quick Find implementation, the time complexity of the `union` function is better. Meanwhile, the `find` function will take more time in this case.



Next, we will learn these two implementations and two common strategies to optimize a disjoint set.



## Quick Find - Disjoint Set

------



### Explanation of Quick Find

------

In this video, we'll talk about Quick Find implementation of a Disjoint Set and cover its two basic operations along with their complexity: find and union.



### Algorithm

------

Here is a sample quick find implementation of the Disjoint Set.

```python
# UnionFind class
class UnionFind:
    def __init__(self, size):
        self.root = [i for i in range(size)]

    def find(self, x):
        return self.root[x]
		
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            for i in range(len(self.root)):
                if self.root[i] == rootY:
                    self.root[i] = rootX

    def connected(self, x, y):
        return self.find(x) == self.find(y)


# Test Case
uf = UnionFind(10)
# 1-2-5-6-7 3-8-9 4
uf.union(1, 2)
uf.union(2, 5)
uf.union(5, 6)
uf.union(6, 7)
uf.union(3, 8)
uf.union(8, 9)
print(uf.connected(1, 5))  # true
print(uf.connected(5, 7))  # true
print(uf.connected(4, 9))  # false
# 1-2-5-6-7 3-8-9-4
uf.union(9, 4)
print(uf.connected(4, 9))  # true
```



### Time Complexity

------

|                     | Union-find Constructor | Find   | Union  | Connected |
| ------------------- | ---------------------- | ------ | ------ | --------- |
| **Time Complexity** | `O(N)`                 | `O(1)` | `O(N)` | `O(1)`    |



Note: `N` is the number of vertices in the graph.

- When initializing a `union-find constructor`, we need to create an array of size `N` with the values equal to the corresponding array indices; this requires linear time.
- Each call to `find` will require `O(1)` time since we are just accessing an element of the array at the given index.
- Each call to `union` will require `O(N)` time because we need to traverse through the entire array and update the root vertices for all the vertices of the set that is going to be merged into another set.
- The `connected` operation takes `O(1)` time since it involves the two `find` calls and the equality check operation.

### Space Complexity

------

We need `O(N)` space to store the array of size `N`.



## Quick Union - Disjoint Set

------



### Explanation of Quick Union

------

In the following video we'll take a look at Quick Union implementation of a Disjoint Set and show the difference between the Quick Union implementation and the Quick Find implementation we talked about earlier. As previously done for the Quick Find implementation, we'll also derive the time complexity of the Quick Union operations so you can compare them.



### Why is Quick Union More Efficient than Quick Find?

------

Generally speaking, Quick Union is more efficient than Quick Find. We'll explain the reason in the below video.



Clarifying Notes



The keen observer may notice that the Quick Union code shown here includes a soon-to-be introduced technique called path compression. While the complexity analysis in the video is sound, it is intended for the implementation of Quick Union that is shown below.

### Algorithm

------

Here is a sample quick union implementation of the Disjoint Set.

```python
# UnionFind class
class UnionFind:
    def __init__(self, size):
        self.root = [i for i in range(size)]

    def find(self, x):
        while x != self.root[x]:
            x = self.root[x]
        return x
    
    # this optimization is discussed in Path Compression Optimization below
    def find_op(self, x):
        """
        there is one better version of find function
        we update root[x] during recursive calls
        for instance, take a look of this path:
        		A -> B -> C -> D
        root[x] A	 A 	  B	   C
        
        after we call find(D), we update root to this:
        		A -> B -> C -> D
        root[x] A	 A 	  A	   A
        
        hence, the next time that we need to find the root of D, it will not call find function four times, instead, it will find A in the second call.
        """
        if self.root[x] != x:
            self.root[x] = find_op(self.root[x])
        return self.root[x]
		
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.root[rootY] = rootX

    def connected(self, x, y):
        return self.find(x) == self.find(y)


# Test Case
uf = UnionFind(10)
# 1-2-5-6-7 3-8-9 4
uf.union(1, 2)
uf.union(2, 5)
uf.union(5, 6)
uf.union(6, 7)
uf.union(3, 8)
uf.union(8, 9)
print(uf.connected(1, 5))  # true
print(uf.connected(5, 7))  # true
print(uf.connected(4, 9))  # false
# 1-2-5-6-7 3-8-9-4
uf.union(9, 4)
print(uf.connected(4, 9))  # true
```

### Time Complexity

------

|                     | Union-find Constructor | Find   | Union  | Connected |
| ------------------- | ---------------------- | ------ | ------ | --------- |
| **Time Complexity** | `O(N)`                 | `O(N)` | `O(N)` | `O(N)`    |



Note: `N` is the number of vertices in the graph. In the worst-case scenario, the number of operations to get the root vertex will be `H` where `H` is the height of the tree. Because this implementation does not always point the root of the shorter tree to the root of the taller tree, `H` can be at most `N` when the tree forms a linked list.

- The same as in the quick find implementation, when initializing a `union-find constructor`, we need to create an array of size `N` with the values equal to the corresponding array indices; this requires linear time.
- For the `find` operation, in the worst-case scenario, we need to traverse every vertex to find the root for the input vertex. The maximum number of operations to get the root vertex would be no more than the tree's height, so it will take `O(N)` time.
- The `union` operation consists of two `find` operations which (**only in the worst-case**) will take `O(N)` time, and two constant time operations, including the equality check and updating the array value at a given index. Therefore, the `union` operation also costs `O(N)` in the worst-case.
- The `connected` operation also takes `O(N)` time in the worst-case since it involves two `find` calls.

### Space Complexity

------

We need `O(N)` space to store the array of size `N`.



## Union by Rank - Disjoint Set

### Disjoint Set - Union by Rank

------

We have implemented two kinds of “disjoint sets” so far, and they both have a concerning inefficiency. Specifically, the quick find implementation will always spend O(n) time on the union operation and in the quick union implementation, as shown in Figure 1, it is possible for all the vertices to form a line after connecting them using `union`, which results in the worst-case scenario for the `find` function. Is there any way to optimize these implementations?

Of course, there is; it is to union by rank. The word “rank” means ordering by specific criteria. Previously, for the `union` function, we always chose the root node of `x` and set it as the new root node for the other vertex. However, by choosing the parent node based on certain criteria (by rank), we can limit the maximum height of each vertex.

To be specific, the “rank” refers to the height of each vertex. When we `union` two vertices, instead of always picking the root of `x` (or `y`, it doesn't matter as long as we're consistent) as the new root node, we choose the root node of the vertex with a larger “rank”. We will merge the shorter tree under the taller tree and assign the root node of the taller tree as the root node for both vertices. In this way, we effectively avoid the possibility of connecting all vertices into a straight line. This optimization is called the “disjoint set” with union by rank.



![img](https://leetcode.com/explore/learn/card/Figures/Graph_Explore/A_Line_Graph.png)



Figure 6. A line graph



### Video Explanation

------

In this video, you'll learn how to actually implement the “disjoint set” with union by rank.



Clarifying Notes



At time 0:52 we have effectively constructed a linked list of nodes with 5 as the root node, where 4 points to 5, 3 to 4, 2 to 3, 1 to 2, and 0 to 1. This demonstrates the main inefficiency of Quick Union. A keen observer will notice that `union(x, y)` in our previous implementation of Quick Union, will always point `rootY` to `rootX`. So the actual order of operations to produce the "tree" shown at 0:52 would be `union(1, 0)`, `union(2, 0)`, `union(3, 0)`, `union(4, 0)`, `union(5, 0)`. Nevertheless, the key idea remains the same: Quick Union runs the risk of forming a skewed tree.



### Algorithm

------

Here is the sample implementation of union by rank.

```python
# UnionFind class
class UnionFind:
    def __init__(self, size):
        self.root = [i for i in range(size)]
        self.rank = [1] * size

    def find(self, x):
        while x != self.root[x]:
            x = self.root[x]
        return x
		
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.root[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.root[rootX] = rootY
            else:
                self.root[rootY] = rootX
                self.rank[rootX] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)


# Test Case
uf = UnionFind(10)
# 1-2-5-6-7 3-8-9 4
uf.union(1, 2)
uf.union(2, 5)
uf.union(5, 6)
uf.union(6, 7)
uf.union(3, 8)
uf.union(8, 9)
print(uf.connected(1, 5))  # true
print(uf.connected(5, 7))  # true
print(uf.connected(4, 9))  # false
# 1-2-5-6-7 3-8-9-4
uf.union(9, 4)
print(uf.connected(4, 9))  # true
```

### Time Complexity

------

|                     | Union-find Constructor | Find      | Union     | Connected |
| ------------------- | ---------------------- | --------- | --------- | --------- |
| **Time Complexity** | `O(N)`                 | `O(logN)` | `O(logN)` | `O(logN)` |



Note: `N` is the number of vertices in the graph.

- For the `union-find` constructor, we need to create two arrays of size `N` each.
- For the `find` operation, in the worst-case scenario, when we repeatedly union components of equal rank, the tree height will be at most `log(N) + 1`, so the `find` operation requires `O(log N)` time.
- For the `union` and `connected` operations, we also need `O(log N)` time since these operations are dominated by the `find` operation.

### Space Complexity

------

We need `O(N)` space to store the array of size `N`.



