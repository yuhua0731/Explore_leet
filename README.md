# Explore_leet
## heapq in python
heap queue, aka priority queue, always keep all elements sorted with smallest element in the first position
### 2 different properties:
- zero-based indexing, h[i]'s children are h[2 * i + 1] and h[2 * i + 2]
- min-heap, pop method returns the smallest element
tip: if you want to implement a max heap, just negative all elements
### common use functions
- initialize: h = [] or h = heapify(list)
- heappush(heap, item)
- heappop(heap)
- heappushpop(heap, item): push item first, then pop the smallest item
- heapreplace(heap, item): pop smallest item first, then push item
- merge(list1, list2)
- merge(*iterables, key=None, reverse=False) do not know how to use...
- nlargest/nsmallest: return a list