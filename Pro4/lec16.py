import matplotlib.pyplot as plt
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from mlinsights.mlmodel import KMeansL1L2

# — — — — — — -Assigning Initial Centers — — — — — — — — — — — -
centers = [[-5, 2], [0, -6]]
# centers = [[0, -6], [-5, 2]]

# — — — — — — -Assigning Data: Dummy Data used in example above — — — — — — — — — — — — — — — — — —
df=np.array([[0, -6], [4, 4], [0, 0], [-5, 2]])

# — — — — — — -Fit KMedoids clustering — — — — — — — — — — — -
# KMobj = KMedoids(n_clusters=2, metric='l1').fit(df)
# # — — — — — — -Assigning Cluster Labels — — — — — — — — — — — -
# labels = KMobj.labels_
# center = KMobj.cluster_centers_

# KMobj = KMeans(n_clusters=2).fit(df)
# labels = KMobj.labels_
# center = KMobj.cluster_centers_

kml1 = KMeansL1L2(2, norm='L1')
kml1.fit(df)
labels = kml1.labels_
center = kml1.cluster_centers_

cluster1_center = center[0]

# Find Cluster 1 Members
cluster1_members = df[labels == 0]

# Find Cluster 2 Center
cluster2_center = center[1]

# Find Cluster 2 Members
cluster2_members = df[labels == 1]

print(labels)
print(center)
print("Cluster 1 Center:", cluster1_center)
print("Cluster 1 Members:", cluster1_members)
print("Cluster 2 Center:", cluster2_center)
print("Cluster 2 Members:", cluster2_members)