import numpy as np


def calcDist(cluster):
    """Get distance of each tweet in cluster agaist the other tweets from same cluster
    """

    dist = [sum([calcJaccard(x, y) for y in cluster]) for x in cluster]

    return dist


def calcCentroid(cluster, k):

    centro = {i: [] for i in range(k)}

    for k_num, tweet_cluster in cluster.items():

        # get the distance of each tweet against each other in one cluster
        distance = calcDist(tweet_cluster)
        # the tweet with the smallest aggregate distance should be considered the new "center"

        centro[k_num] = tweet_cluster[np.argmin(distance)]

    return centro


def calcJaccard(x, y):
    x = set(x)
    y = set(y)

    intersection_len = len(list((x & y)))
    union_len = len(list((x | y)))

    return 1 - (intersection_len/union_len)


def calcSSE(cluster, centro):
    """Get intercluster distance, and sum it all up
    """

    sse = 0

    for k_num, centroid_tweets in centro.items():

        for tweet in cluster[k_num]:

            dist = calcJaccard(centroid_tweets, tweet)**2
            sse = sse + dist

    return round(sse,2)
