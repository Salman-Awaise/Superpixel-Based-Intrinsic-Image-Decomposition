# keeping the command line entry point here
import argparse

from src import config
from src import pipeline


# parsing command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Superpixel-based intrinsic image decomposition on the MIT Intrinsic Images dataset.")
    # writing figures to disk instead of only displaying them
    parser.add_argument("--save-figures", action="store_true",
                        help="also write every figure to results/figures/")
    # skipping all plotting for a headless metrics-only run
    parser.add_argument("--no-figures", action="store_true",
                        help="skip plotting entirely and only print metrics")
    # overriding the number of clusters
    parser.add_argument("--k", type=int, default=None,
                        help=f"number of K-Means clusters (default {config.K})")
    # overriding the SLIC superpixel count
    parser.add_argument("--n-segments", type=int, default=None,
                        help=f"approximate number of superpixels (default {config.N_SEGMENTS})")
    return parser.parse_args()


# running the full analysis
def main():
    args = parse_args()
    # enabling figure saving when requested
    if args.save_figures:
        config.SAVE_FIGURES = True
    # applying parameter overrides
    if args.k is not None:
        config.K = args.k
    if args.n_segments is not None:
        config.N_SEGMENTS = args.n_segments
    pipeline.run_all(show=not args.no_figures)


if __name__ == "__main__":
    main()
