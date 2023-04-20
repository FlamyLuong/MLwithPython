import numpy as np
from scipy.stats import norm

# Define the dataset
D = [-1, 0, 4, 5, 6]

# Define the parameters
theta = [0.5, 0.5, 6, 7, 1, 4]

# Extract the parameters
pi1, pi2, mu1, mu2, sigma1_sq, sigma2_sq = theta

# Compute the log-likelihood
log_likelihood = 0
for x in D:
    likelihood_x = pi1 * norm.pdf(x, mu1, np.sqrt(sigma1_sq)) + pi2 * norm.pdf(x, mu2, np.sqrt(sigma2_sq))
    log_likelihood += np.log(likelihood_x)

print("The log-likelihood of the data given the initial setting of theta is: {:.1f}".format(log_likelihood))

