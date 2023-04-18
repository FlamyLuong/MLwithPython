from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
import numpy as np

# X = np.array([[0, 0],[2, 0],[0, 2],[2, 2],[5, 1], [5, 2], [2, 4],[4, 4]])
# y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

# X = np.array([[2, 0], [3, 0], [0, 2], [2, 2], [5, 1], [5, 2], [2, 4], [4, 4], [5, 5], [0, 0]])
# y = np.array([-1, -1, -1, -1, 1, 1, 1, 1, 1, -1])

# clf = Perceptron(tol=1e-3, random_state=0)
# clf.fit(X, y)
#
# print(clf.coef_)
# print(clf.intercept_)
X = np.array([[0, 0], [2, 0], [3, 0], [0, 2], [2, 2], [5, 1], [5, 2], [2, 4], [4, 4], [5, 5]])
y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])

clf = SVC(kernel="linear", C=1)
clf.fit(X,y)

print(clf.coef_)
w1, w2 = clf.coef_[0] / 2

print(clf.intercept_)
b = clf.intercept_ / 2

# Calculate sum of hinge losses
hinge_losses = []
for i in range(len(X)):
    x = X[i]
    z = y[i]
    hinge_loss = max(0, 1 - z*(w1*x[0] + w2*x[1] + b))
    hinge_losses.append(hinge_loss)

sum_hinge_losses = sum(hinge_losses)

print(sum_hinge_losses)
#
#
#
# import numpy as np
# theta = np.array([0,0]).T
# theta_0 = 0
# x = np.array([[0, 0], [2, 0], [3, 0], [0, 2], [2, 2], [5, 1], [5, 2], [2, 4], [4, 4], [5, 5]])
# y = [-1,-1,-1,-1,-1,1,1,1,1,1]
# per = [1, 9, 10, 5, 9, 11, 0, 3, 1, 1]
# error = 0
# for t in range(15):
#     for i in range(10):
#         if y[i]*(theta@x[i]+theta_0) <= 0:
#             theta=theta+y[i]*x[i]
#             theta_0 = theta_0 + y[i]
#             error += 1
# print(theta, theta_0)
# print(error)
#
# # !/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed Jun 19 12:59:45 2019
# @author: dileepn
# Perceptron algorithm with offset
# """
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Data points
# x = np.array([[0, 0], [2, 0], [3, 0], [0, 2], [2, 2], [5, 1], [5, 2], [2, 4], [4, 4], [5, 5]])
# # x = np.array([[-4,2],[-2,1],[-1,-1],[2,2],[1,-2]])
# # x = np.array([[1,0],[-1,10],[-1,-1]])
#
# # Labels
# y = np.array([[-1], [-1], [-1], [-1], [-1], [1], [1], [1], [1], [1]])
# # y = np.array([[1],[1],[-1],[-1],[-1]])
# # y = np.array([[-1],[1],[1]])
#
# # Plot data
# colors = ['b' if y == 1 else 'r' for y in y]
# plt.figure()
# plt.scatter(x[:, 0], x[:, 1], s=40, c=colors)
#
# # Number of examples
# n = x.shape[0]
#
# # Number of features
# m = x.shape[1]
#
# # No. of iterations
# T = 15
#
# # Initialize parameter vector and offset
# theta = np.array([[1], [1]])
# theta0 = 0
#
# # Tolerance for floating point errors
# eps = 1e-8
#
# # Start the perceptron update loop
# mistakes = 0  # Keep track of mistakes
# for t in range(T):
#     counter = 0  # To check if all examples are classified correctly in loop
#     for i in range(n):
#         agreement = float(y[i] * (theta.T.dot(x[i, :]) + theta0))
#         if abs(agreement) < eps or agreement < 0.0:
#             theta = theta + y[i] * x[i, :].reshape((m, 1))
#             theta0 = theta0 + float(y[i])
#             print("current parameter vector:", theta)
#             print("current offset: {:.1f}".format(theta0))
#             mistakes += 1
#         else:
#             counter += 1
#
#     # If all examples classified correctly, stop
#     if counter == n:
#         print("No. of iteration loops through the dataset:", t + 1)
#         break
#
# # Print total number of mistakes
# print("Total number of misclassifications:", mistakes)
#
# # Plot the decision boundary
# x_line = np.linspace(-1, 6, 100)
# y_line = (-theta0 - theta[0] * x_line) / theta[1]
# y_line2 = (18 - 4 * x_line) / 4
# plt.plot(x_line, y_line, 'k-', linewidth=2, label='Max. Margin Separator')
# plt.plot(x_line, y_line2, 'g--', linewidth=1, label='Perceptron Solution')
# plt.legend()