import argparse
import glob
import itertools
import logging
import os
import pprint
import shlex
import subprocess
import sys
import uuid

from parser import parse_args

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


def run_subprocess(command: str):
    logger.debug(command)

    process = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=2,
    )
    with process.stdout as p_out:
        for line in iter(p_out.readline, b""):  # b'\n'-separated lines
            logger.info(line.decode().strip())

    process.wait()  # to fetch returncode
    return process.returncode


def main(args):
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "exp_" + args.pipeline
    run_id = args.run_id if args.run_id is not None else uuid.uuid4().hex
    run_uuid = uuid.uuid4().hex

    # returncode = run_subprocess(command="ray start --head")
    # if returncode:
    #     raise Exception(f"Failed to start ray cluster. Errorcode: {str(returncode)}")
    try:
        failed_runs, nb_failed_runs = [], 0
        command = " ".join(
            [
                "kedro",
                "run",
                f"--pipeline={args.pipeline}",
                '--params="'
                + f"run_id:{run_id}"
                + ","
                + f"run_uuid:{run_uuid}"
                + '"',
            ]
        )

        returncode = run_subprocess(command)
        if returncode:
            nb_failed_runs += 1
            logger.critical(
                f"Pipeline {args.pipeline} failed with errorcode: {str(returncode)}."
            )
            raise Exception

    except Exception as e:
        run_subprocess("ray stop --force")
        raise Exception(e)
    else:
        run_subprocess("ray stop --force")


args = parse_args(args=sys.argv[1:])

if __name__ == "__main__":
    main(args)
