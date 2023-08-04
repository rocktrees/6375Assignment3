import argparse

from Kmeans import Kmeans
from utils.util import cleanData


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
                        default="https://raw.githubusercontent.com/rocktrees/6375Assignment3/main/reuters_health.txt"
                        )
    parser.add_argument(
        "--log",
        type=bool,
        default=False
    )
    return parser


def main():
    args = get_parser().parse_args()

    procData = cleanData(args.file_path)

    kmeans = Kmeans(k=args.k, tweets=procData)

    kmeans.fit(args.log)


if __name__ == "__main__":
    main()
