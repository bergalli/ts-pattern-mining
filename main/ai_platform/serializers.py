from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

from parser import CLIArgument


class JobRequest(BaseModel):
    run_id: str
    run_uuid: str

    def prepare_job_inputs(self, **dependencies):
        raise NotImplementedError


class CreativeScoreRequest(JobRequest):
    pipeline: str

    def prepare_job_inputs(self, **dependencies):
        args = [
            f"--{CLIArgument.RUN_ID}={self.run_id}",
            f"--{CLIArgument.PIPELINE}={self.pipeline}",
        ]
        return args

    # def _check_all_inputs_defined(self): # todo
    #     #assert all(self.__dict__.values()...)
