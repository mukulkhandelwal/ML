import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans


X = np.array( [ [1,2],[1.5,1.8],[5,8],[8,8],[1,0.5],[9,11]])
#
plt.scatter(X[:,0],X[:,1], s=150, linewidths=5)
plt.show()

# clf = KMeans(n_clusters=2)
# clf.fit(X)
#
# centroids = clf.cluster_centers_
# labels = clf.labels_

colors = ["g","r","c","b","k","o"]

# for i in range(len(X)):
#     plt.plot(X[i][0],X[i][1], colors[labels[i]], markersize = 10)  # labels 2 so green or red
#
# plt.scatter(centroids[:,0], centroids[:,1], marker ='x', s=150,linewidths=5)
# plt.show()
#


class K_Means:
    def __init__(self, k=2, tol = 0.001, max_iter = 300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter


    def fit(self,data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]  #you can also shuffle data for random picking centroids

        for i in range(self.max_iter):
            self.classifications = {}


            for i in range(self.k):
                self.classifications[i] = []


            for featureset in X:  #X is as data
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances)) #finding index
                self.classifications[classification].append(featureset)


            prev_centroids = dict(self.centroids)


            for classification in self.classifications:
                pass
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)





    def predict(self,data):
        pass



