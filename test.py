#!/usr/bin/env python3
from solution import ListNode, TreeNode, fileHandler
import time
import csv
import json
from bitstring import BitArray

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
print(msg[-4:])

a = int.from_bytes(b'\xff', byteorder='little', signed=True)
b = a + 1000
print(a)
print(b)
print(b - a)

print(int(BitArray(hex=b'\x01'.hex()).bin[7]))