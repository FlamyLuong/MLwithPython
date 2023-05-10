import numpy as np

s = np.array([[0, 0]])
A = np.array([[1, 0]])
B = np.array([[0, 1]])

w_ss = np.array([[-1, 0], [0, 1]])
w_sx = np.array([[1, 0], [0, 1]])

x1 = np.array([[1,0],[1,0]])
x2 = np.array([[1,0], [0,1], [0,1]])
x3 = np.array([[0,1], [1,0], [1,0]])


s1 = (w_ss @ s.T) + (w_sx @ B.T)
s11 = (w_ss @ s1) + (w_sx @ A.T)
s12 = (w_ss @ s11) + (w_sx @ A.T)
print(s11)
print(s12)

# s2 = (w_ss * s1) + (w_sx * A * B * B)
# print(s2)
#
# s3 = (w_ss * s2) + (w_sx * A * B * B)
# print(s3)


