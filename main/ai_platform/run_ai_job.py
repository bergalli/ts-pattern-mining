import subprocess
import uuid
from typing import Tuple

from ai_platform.serializers import JobRequest


def run_ai_job(job_config, job_request: JobRequest, job_purpose: str) -> Tuple[bool, bytes]:
    """
    Runs a ai platform training job using a custom docker image from creative-drivers repository
    See ia-creative-drivers/scripts/cli.py for further information
    :param cfg:
    :param job_request:
    :param storage_client:
    :return:
    """
    job_args = [
        "gcloud",
        "ai-platform",
        "jobs",
        "submit",
        "training",
        f"{job_request.run_id.replace('-', '_')}__{job_request.run_uuid.replace('-', '_')}",
        f"--region={job_config.region}",
        f"--scale-tier={job_config.scaleTier}",
        f"--master-machine-type={job_config.masterType}",
        f"--master-image-uri={job_config.masterImageUri}",
        "--",
    ] + job_request.prepare_job_inputs()
    result = subprocess.run(args=job_args, capture_output=True, check=False)

    if result.returncode:
        return False, result.stderr

    return True, bytes()
