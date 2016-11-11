import numpy as np
import amitgroup as ag

X = np.array(
    [[[0, 0, 1, 0, 1],
      [1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]]] * 4, dtype=np.uint8)

X = np.ones((174, 15, 15, 4), dtype=np.uint8)

for i in range(1000):
    Xs = ag.features.bspread(X, spread='orthogonal', radius=3)

print "Before"
print X
print "After"
print Xs
