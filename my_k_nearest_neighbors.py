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

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k :
        warnings.warn(' K is set to a value less than total voting groups !')

    distances = []
    for group in data:
        for features in data[group]:
            #euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict)) **2)) #same as euclidean distance but fast than that formula and can have more than 2 dimension
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))  #same as above bcz it is faster
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]] #finding groups
    print(Counter(votes).most_common(1))
    #
    # for i in sorted(distances)[:k]:  #same as above
    #     i[i]

    vote_result = Counter(votes).most_common(1)[0][0]  #1 means only numbers=1 common group we want  ,0 0 most common group and how many

    return vote_result


result = k_nearest_neighbors(dataset ,new_feature, k=3)
print(result)

[[plt.scatter(ii[0], ii[1], s=100, color = i)for ii in dataset[i]] for i in dataset]
plt.scatter(new_feature[0], new_feature[1], color = result)
plt.show()

