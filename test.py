

import numpy as np

lines = []
with open("data/brown-test.txt", 'r') as f:
    for i, line in enumerate(f):
        # if line != '\n':
            # lines.append(line.split())
        print(line)
        print('-'*50)
        if i == 50:
            break

# print(len(lines))