import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

# Load data
data = pd.read_csv(r'C:\Users\flamy\PycharmProjects\pythonProject\data_analysis\data_and_materials\gamma-ray.csv')
gamma_data  = pd.DataFrame(data)
# print(df)

#part d
lambda_hat_H0 = gamma_data['count'].sum()/gamma_data['seconds'].sum()
lambdas_hat_H1 = gamma_data['count']/gamma_data['seconds']
print("H0 = {}".format(lambda_hat_H0))
print("H1 = {}".format(lambdas_hat_H1))

#part f-2

plot_Xs = np.arange(50,150,0.1)
plt.plot(plot_Xs, scipy.stats.chi2.pdf(plot_Xs, 99))
plt.show()

#part f-3
chi = scipy.stats.chi2.isf(0.05, 99)
# print(chi)

#part f-4
def likelihood_H0(lamb):
  # The likelihood function is a product of Poisson distributions. For H0, each Poisson distribution
  # has the same mean.
  return scipy.stats.poisson.pmf(gamma_data['count'], gamma_data['seconds']*lamb).prod(axis=0)

def likelihood_H1(lambs):
  # For H1, the means for the Poisson distributions are given by the parameter 'lambs'
  return scipy.stats.poisson.pmf(gamma_data['count'], gamma_data['seconds']*lambs).prod(axis=0)

# The test statistic for the MLE is given by calculating the likelihood ratio for the MLE estimators calculated earlier.
Lambda_observed = -2*np.log(likelihood_H0(lambda_hat_H0)/likelihood_H1(lambdas_hat_H1))

# with the MLE estimators.
pvalue = scipy.stats.chi2.sf(Lambda_observed, 99)
print(Lambda_observed, pvalue)