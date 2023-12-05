import cProfile

def bubble_sort(x):
    n = len(x)
    for i in range(n):
        # 最后i个元素已经排好，无需再比较
        for j in range(0, n-i-1):
            # 从第一个到倒数第i+1个元素，两两比较并交换
            if x[j] > x[j+1]:
                x[j], x[j+1] = x[j+1], x[j]
    return x

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result

def merge_sort(x):
    if len(x) <= 1:
        return x

    # 将列表分成两半
    mid = len(x) // 2

    # 递归地对两半进行排序
    left = merge_sort(x[:mid])
    right = merge_sort(x[mid:])

    # 合并两个有序的子列表
    return merge(left, right)

def quick_sort(x):
    if len(x) <= 1:
        return x

    pivot = x[len(x) // 2]  # 选择基准元素
    left = [x for x in x if x < pivot]  # 小于基准的元素
    middle = [x for x in x if x == pivot]  # 等于基准的元素
    right = [x for x in x if x > pivot]  # 大于基准的元素

    return quick_sort(left) + middle + quick_sort(right)

def insertion_sort(x):
    for i in range(1, len(x)):
        key = x[i]
        j = i - 1
        while j >= 0 and key < x[j]:
            x[j + 1] = x[j]
            j -= 1
        x[j + 1] = key

def selection_sort(x):
    n = len(x)
    for i in range(n):
        min_index = i
        for j in range(i+1, n):
            if x[j] < x[min_index]:
                min_index = j
        x[i], x[min_index] = x[min_index], x[i]

def general_sort(x):
    return sorted(x)

globals_dict = globals()
locals_dict = {'x': [i for i in range(5000, 0, -1)]}

print("bubble_sort")
cProfile.runctx('bubble_sort(x)', globals_dict, locals_dict)
print("merge_sort")
cProfile.runctx('merge_sort(x)', globals_dict, locals_dict)
print("quick_sort")
cProfile.runctx('quick_sort(x)', globals_dict, locals_dict)
print("insertion_sort")
cProfile.runctx('insertion_sort(x)', globals_dict, locals_dict)
print("selection_sort")
cProfile.runctx('selection_sort(x)', globals_dict, locals_dict)
print("general_sort")
cProfile.runctx('general_sort(x)', globals_dict, locals_dict)