import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from math import sqrt
import warnings
from collections import Counter

# plot1 = [1,3]
# plot2 = [2,5]
#euclidean_distance = sqrt( ( plot1[0]-plot2[0])** 2 + (plot1[1] - plot2[1]) ** 2)
#print(euclidean_distance)


style.use('fivethirtyeight')

dataset = {'k' : [[1,2] ,[2,3], [3,1]], 'r' : [[6,5], [7,7], [8,6]] }
new_feature = [5,7]

#
# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], s=100, color = i) #plt.scatter()
#
#above can be do as

# [[plt.scatter(ii[0], ii[1], s=100, color = i)for ii in dataset[i]] for i in dataset]
# plt.scatter(new_feature[0], new_feature[1])
# plt.show()

def k_nearest_neighbors(data, predict, k=3):
    if len(data) <= k :
        warnings.warn(' K is set to a valye less than total voting groups !')

    # knnalgos
    # return vote_result



