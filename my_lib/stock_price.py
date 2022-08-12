#!/usr/bin/env python3
import collections
import heapq

class StockPrice:

    def __init__(self):
        self.price = collections.defaultdict(int)
        self.latest = 0
        self.max_pq = []
        self.min_pq = []

    def update(self, timestamp: int, price: int) -> None:
        self.price[timestamp] = price
        self.latest = max(self.latest, timestamp)
        heapq.heappush(self.min_pq, (price, timestamp))
        heapq.heappush(self.max_pq, (-price, timestamp))

    def current(self) -> int:
        return self.price[self.latest]

    def maximum(self) -> int:
        while -self.max_pq[0][0] != self.price[self.max_pq[0][1]]:
            heapq.heappop(self.max_pq)
        return -self.max_pq[0][0]

    def minimum(self) -> int:
        while self.min_pq[0][0] != self.price[self.min_pq[0][1]]:
            heapq.heappop(self.min_pq)
        return self.min_pq[0][0]

if __name__ == '__main__':
    # Your StockPrice object will be instantiated and called as such:
    obj = StockPrice()
    obj.update(1, 3)
    print(obj.current())
    obj.update(3, 5)
    obj.update(1, 7)
    print(obj.maximum())
    print(obj.minimum())