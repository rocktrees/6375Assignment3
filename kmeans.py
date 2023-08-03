from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from utils.calc import calcCentroid, calcJaccard, calcSSE

import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k",
        type=int,
        help=("Number of clusters"),
        default=10
    )
    parser.add_argument("--file_path",
                        type=str,
                        default="https://raw.githubusercontent.com/rocktrees/assignment3/main/reuters_health.txt"
                        )
    return parser


def preProc(file_path):

    df = pd.read_csv(file_path, header=None, sep="|")
    tweets = list(df[2])

    cleaned = [tweetFix(tweet) for tweet in tweets]
    return cleaned


def tweetFix(tweet):
    s = re.sub(r'@\w+|http://\S+', '', tweet)
    s = re.sub(r'RT.*?:', '', s)
    s = s.replace(":", "")
    s = s.replace("#", "")
    s = s.lower()

    return s


class Kmeans:
    def __init__(self, k: int, tweets: List[str]):
        self.k = k
        self.tweets = tweets
        self.converge = False
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

    def fit(self):
        print("Converging")
        clusters = []
        while self.converge == False:
            clusters = self.clustering()
            new_centroids = calcCentroid(clusters, self.k)
            self.converge = self.convgCheck(new_centroids)

        print("--------------------------")
        print("Covergence Completed")
        calcSSE_result = calcSSE(clusters, self.centroids)

        print("Sum Squared Error Result Is Shown Below ->")
        print(calcSSE_result)

        for i in range(self.k):
            print(
                f"Total Number Of Tweets In Cluster {(i+1)} -> {len(clusters[i])}")


def main():
    args = get_parser().parse_args()

    procData = preProc(args.file_path)

    # newKmeans(procData, k=args.k, centro=None)
    kmeans = Kmeans(k=args.k, tweets=procData)

    kmeans.fit()


if __name__ == "__main__":
    main()
