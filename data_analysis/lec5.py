import numpy as np
from numpy import linalg as LA

# x = np.array([[0,2], [0,-2], [1,1], [-1, -1]])
#
# # print(x)
# # print(np.cov(x.T))
#
# S = np.cov(x.T)
# eigenvalues, eigenvectors = LA.eigh(S)
# print(eigenvalues)
# print(eigenvectors)


X = np.array([[1, 1], [1, -1], [-1, 1]])
B = np.dot(X, X.T)
print(B)

eigvals, eigvecs = np.linalg.eig(B)
# principal_components = eigvecs[:, :2]
# print(principal_components)

# # Sort the eigenvalues in decreasing order
eigvals = eigvals[::-1]
#
# # Sort the eigenvectors accordingly
# eigvecs = eigvecs[:, ::-1]

# Print the eigenvalues
print(eigvals)

# Print the eigenvectors
print(eigvecs)

v_lambda_2 = eigvecs[:, 1]
v_lambda_2 = v_lambda_2 / np.linalg.norm(v_lambda_2)

# Print the eigenvector
print(v_lambda_2)