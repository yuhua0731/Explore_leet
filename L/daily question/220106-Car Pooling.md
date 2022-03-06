There is a car with `capacity` empty seats. The vehicle only drives east (i.e., it cannot turn around and drive west).

You are given the integer `capacity` and an array `trips` where `trip[i] = [numPassengersi, fromi, toi]` indicates that the `ith` trip has `numPassengersi` passengers and the locations to pick them up and drop them off are `fromi` and `toi` respectively. The locations are given as the number of kilometers due east from the car's initial location.

Return `true` *if it is possible to pick up and drop off all passengers for all the given trips, or* `false` *otherwise*.

 

**Example 1:**

```
Input: trips = [[2,1,5],[3,3,7]], capacity = 4
Output: false
```

**Example 2:**

```
Input: trips = [[2,1,5],[3,3,7]], capacity = 5
Output: true
```

 

**Constraints:**

- `1 <= trips.length <= 1000`
- `trips[i].length == 3`
- `1 <= numPassengersi <= 100`
- `0 <= fromi < toi <= 1000`
- `1 <= capacity <= 10 ** 5`

#### My Approach

> 1. two heap queues: 
>
>    - one(==remain==) stores all trips that have not been visited yet, element = [fromi, toi, numPassengersi]; 
>
>    - the other(==travel==) stores trips that are currently on going, element = [toi, numPassengersi].
>
> 2. in each loop, pop the smallest element from remain, and more elements if they share the same ==fromi== value with the smallest one. Also pop out all elements from travel if toi is less than or equal to fromi.
>
> 3. Check if passenger amount is exceed car’s capacity. If yes, return False
>
> 4. return True at the end.

```python
def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
    remain = sorted([[fromi, toi, numi] for numi, fromi, toi in trips])
    travel = []
    cnt = 0

    while remain:
        fromi = remain[0][0]
        while travel and travel[0][0] <= fromi:
            _, gone = heapq.heappop(travel)
            cnt -= gone

        while remain and remain[0][0] <= fromi:
            _, toi, numi = heapq.heappop(remain)
            heapq.heappush(travel, [toi, numi])
            cnt +=  numi
        print(cnt, fromi, remain, travel)
        if cnt > capacity: return False
    return True
```

#### Lee’s idea

**Intuition**



Same as [253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/discuss/278270/Java-Sort-All-Time-Point).
Track the change of capacity in time order.



**Explanation**

1. Save all time points and the change on current `capacity`
2. Sort all the changes on the key of time points.
3. Track the current `capacity` and return `false` if negative

**Complexity**



Time `O(NlogN)`
Space `O(N)`

```python
def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
    for _, change in sorted([x for n, f, t in trips for x in [[f, n], [t, -n]]]):
        capacity -= change
        if capacity < 0: return False
    return True
```

Jesus..

