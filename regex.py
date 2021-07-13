#!/usr/bin/env python3

import re

m = re.search('[1-9][0-9]*', '12301\ndsa\n1')
print(m.group(0))

test = ['.', '..', '.....', '.p.']
for i in test:
    print(re.match(r'\.+', i).group())