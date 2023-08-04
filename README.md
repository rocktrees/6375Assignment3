# Authors

Hyun Guk Yoo, Kim Le
hxy170003, kml190002

# CS 6375 Assignment 3

### Instructions

please run in command line.

1. For a default run **WITHOUT** saving log, use `python kmeans.py`
   1. This defaults to `k=10` where k is the number of clusters
2. If you want to try a different k, then use the flag `--k <# of clusters>`
   1. ex: `python kmeans.py --k 5`
3. If you would like to keep a `log.txt`, then use the flag `--log True`
   1. This will run k times. Once for each k to 1
   2. For each run, it will save the `sse` and `cluster breakdown` in a text file called `log.txt`
   3. ex: `python kmeans.py --k 5 --log True`
