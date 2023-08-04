# Authors

Hyun Guk Yoo, Kim Le

hxy170003, kml190002

# CS 6375 Assignment 3

### libraries used

1. argparse
2. pathlib
3. re
4. pandas
5. numpy
6. matplotlib.pyplot

### Instructions

please run in command line.

1. For a default run **WITHOUT** saving log, use `python run.py`
   1. This defaults to `k=10` where k is the number of clusters
2. If you want to try a different k, then use the flag `--k <# of clusters>`
   1. ex: `python run.py --k 5`
3. If you would like to keep a `log.txt`, then use the flag `--log True`
   1. This will run k times. Once for each k to 1
   2. For each run, it will save the `sse` and `cluster breakdown` in a text file called `log.txt`
   3. At the end, it will save a plot of the SSE vs Clusters onto the current directory
      1. For the elbow method
   4. ex: `python run.py --k 5 --log True`
