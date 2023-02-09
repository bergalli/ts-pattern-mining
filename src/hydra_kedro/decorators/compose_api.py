from typing import Any, Optional

_UNSPECIFIED_: Any = object()

import functools
import inspect
import os
import shlex
from typing import Callable, List

import hydra
from hydra.core.global_hydra import GlobalHydra
from kedro.framework.session import get_current_session
from .catalog_update import update_catalog_extension

def wrap(config_path: Optional[str] = _UNSPECIFIED_,
         config_name: Optional[str] = None,
         overrides: List[str] = [],
         job_name: Optional[str] = None,
         caller_stack_depth: int = 1,
         return_hydra_config: bool = False):
    # fetch the extra parameters given to kedro via the --params flag

    try:
        kedro_session = get_current_session()
    except:
        kedro_session = None
    if kedro_session:
        kedro_context = kedro_session.load_context()
        kedro_extra_params = kedro_context.params

        if 'hydra' in kedro_extra_params:
            overrides = shlex.split(kedro_extra_params['hydra'])

    def root_decorator(task_fun: Callable):
        taskfun_filepath = inspect.getfile(task_fun)
        taskfun_path = os.path.split(taskfun_filepath)[0]

        decorator_filepath = os.path.realpath(__file__)
        decorator_folder = os.path.split(decorator_filepath)[0]

        config_realpath = os.path.realpath(os.path.join(taskfun_path, config_path))
        config_relpath = os.path.relpath(path=config_realpath, start=decorator_folder)

        if not GlobalHydra.instance().is_initialized():
            hydra.initialize(config_path=config_relpath, job_name=job_name, caller_stack_depth=caller_stack_depth)

        cfg = hydra.compose(config_name=config_name, overrides=overrides, return_hydra_config=return_hydra_config)

        @functools.wraps(task_fun)
        def task_function_execution(**kwargs):
            update_catalog_extension(cfg)
            pipelines_registered = task_fun(cfg, **kwargs)
            return pipelines_registered

        return task_function_execution

    return root_decorator
