from typing import List
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.calc import calcCentroid, calcJaccard, calcSSE
from utils.util import logClusters

class Kmeans:
    def __init__(self, k: int, tweets: List[str]):
        self.k = k
        self.tweets = tweets
        self.converge = False
        self.sse_save = []
        self.clusters = []
        self.initialCentroids()

    def initialCentroids(self):
        """Randomly initialize cluster
        """
        init_centroids = set()

        # prevents same number from being drawn
        while len(init_centroids) < self.k:
            init_centroids.add(np.random.randint(len(self.tweets), size=1)[0])

        centroids = {i: self.tweets[init_centroids.pop()]
                     for i in range(self.k)}

        self.centroids = centroids

    def clustering(self):
        clusters = {i: [] for i in range(self.k)}

        for tweet in self.tweets:

            distance = [calcJaccard(tweet, value)
                        for value in list(self.centroids.values())]

            # find closest centroid
            minimum_dist_idx = np.argmin(distance)

            clusters[minimum_dist_idx].append(tweet)

        return clusters

    def convgCheck(self, new_centroid):
        """Check if centroid changed or not
        """

        for kdx, centroid_tweet in self.centroids.items():
            # even if one old centroid doesnt match the new one, it did not converge
            if centroid_tweet != new_centroid[kdx]:
                self.centroids = new_centroid
                return False

        # if we made it through the loop, it means all centroids did not change!
        return True
    
    def alg(self):
        clusters = 0
        while self.converge == False:
            clusters = self.clustering()
            new_centroids = calcCentroid(clusters, self.k)
            self.converge = self.convgCheck(new_centroids)

        sse = calcSSE(clusters, self.centroids)
        return clusters, sse
    
    def reset(self):
        self.k -= 1
        self.initialCentroids()
        self.converge = False

    def plot(self):
    
        plt.title("Elbow method")
        plt.xlabel("Cluster number")
        plt.ylabel("SSE")

        plt.plot(self.sse_save[::-1])
        plt.savefig("elbow_plot.png")


    def fit(self, log:bool):

        if log:
            for i in range(0, self.k):
                print(f"Converging K: {self.k}")
                clusters, sse = self.alg()

                self.sse_save.append(logClusters(i, self.k, sse, clusters))
                self.reset()
            # only save it at the end. So it can be analyized later
            self.clusters = clusters
            self.plot()

        else:
            print(f"Converging K: {self.k}")
            clusters, sse = self.alg()
            self.clusters = clusters

            print("--------------------------")
            print("Covergence Completed")
            print(f"SSE: {sse}")

            for i in range(self.k):
                print(f"# of tweets in cluster {(i+1)}: {len(clusters[i])}")