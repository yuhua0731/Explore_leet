#!/usr/bin/env python3
from typing import List
import collections
from sortedcontainers import SortedList

class FoodRatings:

    def __init__(self, foods: List[str], cuisines: List[str], ratings: List[int]):
        # current ratings and cuisines
        self.d = dict()

        # group types
        self.types = collections.defaultdict(SortedList)
        for f, c, r in zip(foods, cuisines, ratings):
            self.d[f] = (c, r)
            self.types[c].add((-r, f))

    def changeRating(self, food: str, newRating: int) -> None:
        cuisine, old_rating = self.d[food]
        self.d[food] = (newRating, cuisine)
        self.types[cuisine].remove((-old_rating, food))
        self.types[cuisine].add((-newRating, food))

    def highestRated(self, cuisine: str) -> str:
        return self.types[cuisine][0][1]


if __name__ == '__main__':
    # Your FoodRatings object will be instantiated and called as such:
    # obj = FoodRatings(foods, cuisines, ratings)
    # obj.changeRating(food,newRating)
    # param_2 = obj.highestRated(cuisine)
    in1 = ["FoodRatings", "highestRated", "highestRated", "changeRating", "highestRated", "changeRating", "highestRated"]
    in2 = [[["kimchi", "miso", "sushi", "moussaka", "ramen", "bulgogi"], ["korean", "japanese", "japanese", "greek", "japanese", "korean"], [9, 12, 8, 15, 14, 7]], ["korean"], ["japanese"], ["sushi", 16], ["japanese"], ["ramen", 16], ["japanese"]]
    fr = FoodRatings(['kimchi', 'miso', 'sushi', 'moussaka', 'ramen', 'bulgogi'],['korean', 'japanese', 'japanese', 'greek', 'japanese', 'korean'],[9, 12, 8, 15, 14, 7])
    for function, para in zip(in1[1:], in2[1:]):
        para = str(para)
        para = para[1 : len(para) - 1]
        print(eval('fr.' + function + '(' + para + ')'))