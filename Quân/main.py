import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

K = [1,2,3,4]
seed = [0,1,2,3,4]

# for k in K:
#     min_cost = np.inf
#     for s in seed:
#         mix_init, post_init = common.init(X,k,s)
#         mix, post, cost = kmeans.run(X, mix_init, post_init)
#         if cost < min_cost:
#             min_cost = cost
#     print('Using KMeans, min cost for K = {0} is: {1:.4f}'.format(k,min_cost))
#     common.plot(X, mix, post, 'K-means K = {0}'.format(k))
#
#
# for k in K:
#     min_loglh = np.inf
#     for s in seed:
#         mix_init, post_init = common.init(X,k,s)
#         mix, post, loglh = naive_em.run(X, mix_init, post_init)
#         if loglh < min_loglh:
#             min_loglh = loglh
#
#     print('Using Naive_EM, the log likelihood for K = {0} is: {1:.4f}'.format(k,min_loglh))
#     common.plot(X, mix, post, 'Naive_EM K = {0}'.format(k))

max_BIC = -np.inf
max_K = 0
for k in K:
    for s in seed:
        mix_init, post_init = common.init(X,k,s)
        mix, post, loglh = naive_em.run(X, mix_init, post_init)
        BIC = common.bic(X,mix,loglh)
        if BIC > max_BIC:
            max_BIC = BIC
            max_K = k
print('Max BIC = {0} at k = {1}'.format(max_BIC, max_K))
