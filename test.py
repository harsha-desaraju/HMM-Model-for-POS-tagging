

import numpy as np

lines = []
with open("brown-test.txt", 'r') as f:
    for line in f:
        if line != '\n':
            lines.append(line.split())

print(len(lines))