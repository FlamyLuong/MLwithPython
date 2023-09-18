import zipfile
import numpy as np
import pandas as pd
import numpy.lib.recfunctions
import matplotlib.pyplot as plt
import scipy.stats

# Load data
with zipfile.ZipFile("statsreview_release1.zip") as zip_file: #file python phải để chung folder với file zip
  golub_data, golub_classnames = ( np.genfromtxt(zip_file.open('data_and_materials/golub_data/{}'.format(fname)), delimiter=',', names=True, converters={0: lambda s: int(s.strip(b'"'))}) for fname in ['golub.csv', 'golub_cl.csv'] )

ALL_names, AML_names = (['V{}'.format(int(v)) for v in golub_classnames[golub_classnames['x'] == c][golub_classnames.dtype.names[0]]] for c in[0, 1])
ALL_genes, AML_genes, ALLAML_genes = (numpy.lib.recfunctions.repack_fields(golub_data[names]).view((golub_data[names].dtype[0], len(names))) for names in [ALL_names, AML_names, ALL_names + AML_names])
ALL_N, AML_N, ALLAML_N = (a.shape[1] for a in [ALL_genes, AML_genes, ALLAML_genes])
N_genes = ALLAML_genes.shape[0]

# print(N_genes)

#part a
# Calculate the Welch's t-test statistic
test_statistic = (ALL_genes.mean(axis=1) - AML_genes.mean(axis=1))/np.sqrt(ALL_genes.var(axis=1, ddof=1)/ALL_N + AML_genes.var(axis=1, ddof=1)/AML_N)
# Find the number of degrees of freedom of the statistic according to the approximation formula
test_dof = (ALL_genes.var(axis=1, ddof=1)/ALL_N + AML_genes.var(axis=1, ddof=1)/AML_N)**2 / ( (ALL_genes.var(axis=1, ddof=1)/ALL_N)**2/(ALL_N-1) + (AML_genes.var(axis=1, ddof=1)/AML_N)**2/(AML_N-1))
# Find the 2-sided p-values
p_values = scipy.stats.t.sf(np.abs(test_statistic), test_dof)*2
# Count how many of these p-values are below the significance threshold
(p_values < 0.05).sum()
# print((p_values < 0.05).sum())

#part b
# Sort the p-values in ascending order
p_values_sorted = np.sort(p_values)
# These are the adjusted significance thresholds as an array.
# Each element is the threshold for the corresponding p-value in p_values_sorted
# Note that (np.arange(N_genes)+1)[::-1] gives an array with [N_genes, N_genes-1, N_genes-2, ..., 1]
# The [::-1] reverses the array.
holm_bonferroni_thresholds = 0.05/(np.arange(N_genes)+1)[::-1]
# First we compare the p-values to the associated thresholds. We then get an array
# where the p-values that exceed the threhold have a value of False.
holm_bonferroni_significant = p_values_sorted < holm_bonferroni_thresholds
# We want to find the first value of False in this array (first p-value that exceeds the threshold)
# so we invert it using logical_not.
holm_bonferroni_not_significant = np.logical_not(holm_bonferroni_significant)
# argwhere will return an array of indices for values of True in the supplied array.
# Taking the first element of this array gives the first value of True in holm_bonferroni_not_significant
# which is the same as the first value of False in holm_bonferroni_significant
holm_bonferroni_first_not_significant = np.argwhere(holm_bonferroni_not_significant)[0]
# We reject all hypothesis before the first p-value that exceeds the significance threshold.
# The number of these rejections is exactly equal to the index of the first value that
# exceeds the threshold.
num_holm_bonferroni_rejections = holm_bonferroni_first_not_significant
print(' Holm-Bonferroni: ', num_holm_bonferroni_rejections)



# These are the adjusted significance thresholds as an array.
benjamini_hochberg_thresholds = 0.05*(np.arange(N_genes)+1)/N_genes
# First we compare the p-values to the associated thresholds.
benjamini_hochberg_significant = p_values_sorted < benjamini_hochberg_thresholds
# We are intested in the last p-value which is significant.
# Remeber that argwhere returns an array of indicies for the True values, so
# we take the last element in order to get the index of the last p-value which
# is significant.
benjamini_hochberg_last_significant = np.argwhere(p_values_sorted < benjamini_hochberg_thresholds)[-1]
# We reject all hypotheses before the last significant p-value, AND we reject
# the hypothesis for the last significant p-value as well. So the number of rejected
# hypotheses is equal to the index of the last significant p-value PLUS one.
num_benjamini_hochberg_rejections = benjamini_hochberg_last_significant + 1
print('Benjamini-Hochberg: ', num_benjamini_hochberg_rejections)

