import numpy as np
from numpy import linalg as LA

A = np.array([[3, 0], [0.5, 2]])
v = np.array([[2, 1]])
w = np.array([[0, 1]])

print(LA.eig(A))

