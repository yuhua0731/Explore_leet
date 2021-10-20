#!/usr/bin/env python3
from solution import ListNode, TreeNode, fileHandler
import time
import csv
import json
from bitstring import BitArray
import math
import threading

a = b = list()
a.append(1)
print(a)
print(b)

a = b = 0
c = d = 1
a, b = c, a 
# how this same line value update works?
# 1. get all variables' values at the right sides
# 2. assign to coordinate left variables
print(a, b)
a = c
b = a
print(a, b)

cur = ListNode(1, next=ListNode(2))
head = pre = ListNode(0)
# head is the real leader, never got changed, a pointer which point to the first node
# head.next is where we store our result

head.next = l = cur # pre.next = cur as well
temp = l.next
temp.next = l
l.next = None
pre.next = temp
pre = l

# cur.next, pre, cur= pre, cur, cur.next
ListNode.printList(head)
ListNode.printList(pre)

# a = fileHandler()
# a.writer()
# a.take_writer()

print(ord('z') - ord('A'))

aa = {}
aa["name"] = "name"
print(aa)

strtest = "/+/response"
print(strtest.replace('+', "777777"))

msg = b'\x11\x85\x06\x09\x78\x00\x01\x02\x6c\xff\x87\xb2'
print(msg[-4:].hex())

a = int.from_bytes(b'\xff', byteorder='little', signed=True)
b = a + 1000
print(a)
print(b)
print(b - a)

print(int(BitArray(b'\x01').bin[7]))

rectangles = [[3, 3, 4, 10], [0, 0, 9, 10]]
xs = [x for x1, y1, x2, y2 in rectangles for x in [x1, x2]]
print(xs)

print(math.sqrt(4) is int)
print(math.ceil(5))

temp = [[1, 2], [3, 4]]
print(sum([sum(i) for i in temp]))

a = [1, 2]
b = list()
b.append(a.copy())
a[0] = 0
print(b)

temp = dict()
temp[1] = [2, 3]
for key, [num1, num2] in temp.items():
    print(key, num1, num2)

def periodical_func():
    print(time.time())
    
    threading.Timer(1, periodical_func).start()
    time.sleep(2)
    print("sleep over")

# periodical_func()

a = [9, 9]
a[0] += 1
print(a)

print(max(1, 2, 3, 4))
print('l' in ['l', 'o'])

bigger = 5 > 3
print(not bigger)

eval("0 / 0")