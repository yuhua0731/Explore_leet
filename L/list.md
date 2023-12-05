# list

>  阅读材料：http://www.laurentluce.com/posts/python-list-implementation/

- list in python is a dynamic array

- 当使用insert进行元素插入时，其他元素的id将保持不变

```python
a = [1000, 1001, 1002]
for i in range(len(a)):
    print(id(a[i]))

a.insert(1, 1003)
print(a)
for i in range(len(a)):
    print(id(a[i]))
    
""" output
4574061616
4574061296
4574061584
[1000, 1003, 1001, 1002]
4574061616
4574060944
4574061296
4574061584
"""
```

- list操作的时间复杂度测试

```python
def test_insert_front():
    a = [100000, 100001, 100002]
    for i in range(100000):
        a.insert(0, 100003)

def test_insert_rear():
    a = [100000, 100001, 100002]
    idx = 3
    for i in range(100000):
        a.insert(idx, 100003)
        idx += 1

def test_append():
    a = [100000, 100001, 100002]
    for i in range(100000):
        a.append(100003)

import cProfile
cProfile.run('test_insert_front()')
cProfile.run('test_insert_rear()')
cProfile.run('test_append()')
```

- Insert插入方法的时间复杂度为O(n)【注意：最坏情况下，即插入的位置在列表的起始位置】
- Append插入方法的时间复杂度为O(1)
- Insert至列表的末位，所需要的时间与append基本相同，时间复杂度为O(1)
- Pop O(1)
- Remove O(n)

```
         100004 function calls in 3.001 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.079    0.079    3.001    3.001 2871629878.py:11(test_insert_front)
        1    0.000    0.000    3.001    3.001 <string>:1(<module>)
        1    0.000    0.000    3.001    3.001 {built-in method builtins.exec}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
   100000    2.922    0.000    2.922    0.000 {method 'insert' of 'list' objects}


         100004 function calls in 0.051 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.040    0.040    0.051    0.051 2871629878.py:16(test_insert_rear)
        1    0.000    0.000    0.051    0.051 <string>:1(<module>)
        1    0.000    0.000    0.051    0.051 {built-in method builtins.exec}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
   100000    0.011    0.000    0.011    0.000 {method 'insert' of 'list' objects}


         100004 function calls in 0.045 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.034    0.034    0.045    0.045 2871629878.py:23(test_append)
        1    0.000    0.000    0.045    0.045 <string>:1(<module>)
        1    0.000    0.000    0.045    0.045 {built-in method builtins.exec}
   100000    0.011    0.000    0.011    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```

