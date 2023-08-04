from pathlib import Path
import re
import pandas as pd

def cleanData(file_path):
    
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

def logClusters(i, k, sse, clusters):

    if i == 0:
        # if log_file already exists, then delete it
        log_file = Path("log.txt")
        if log_file.is_file():
            log_file.unlink()

        with open("log.txt", "a") as file:
            file.write(f"Kmeans log\n")
            file.write(f"--------------------------------------------------\n")
            file.write(f"--------------------------------------------------\n")

    with open("log.txt", "a") as file:
        file.write(f"\nK: {k}\tSSE: {sse}\n")
        file.write(f"Size of each cluster\n")
        for i in range(k):
            file.write(f"\tcluster {(i+1)}: {len(clusters[i])} tweets\n")