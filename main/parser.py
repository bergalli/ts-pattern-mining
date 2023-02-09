from enum import Enum
import argparse
import sys


class CLIArgument(str, Enum):
    PIPELINE = "pipeline"
    RUN_ID = "run-id"


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Creative Score command-line interface",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        f"--{CLIArgument.PIPELINE}",
        type=str,
        default="__default__",
    )
    parser.add_argument(f"--{CLIArgument.RUN_ID}", type=str, default=None)

    return parser.parse_args(args)
