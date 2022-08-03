#!/usr/bin/env python3
import numpy as np
 
N = 1000000
p1 = 79.5 + np.random.randn(N, 1) * 2.8
p2 = 77.2 + np.random.randn(N, 1) * 2.9
cnt = 0
for i, j in zip(p1, p2):
    if i < j: cnt += 1
print('{:.1f}% of the trials had p2 > p1'.format(100 * cnt / N))
