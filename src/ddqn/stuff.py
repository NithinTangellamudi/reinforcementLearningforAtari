import numpy as np
from numpy.linalg import matrix_rank
from numpy.linalg import matrix_power

A = np.array([[2, 5], [0, 1]])
C = np.array([0, 1])

obs = np.array([[C], [C@A@A]])
print(np.shape(obs))
print(obs)
r = matrix_rank(obs)
print(r)