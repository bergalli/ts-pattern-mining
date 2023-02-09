import functools
import inspect
import os
import shlex
import sys
from copy import deepcopy
from typing import Optional

import hydra
from hydra.main import _UNSPECIFIED_
from hydra.types import TaskFunction
from kedro.framework.session import get_current_session
from omegaconf import DictConfig

from .catalog_update import update_catalog_extension

# A function decorated with hydra.main can only return None.
# This global variable is used to get a return anyway when a function
#  is decorated with the decorator in this script.
_pipelines_registered = None


def wrap(
        config_path: Optional[str] = _UNSPECIFIED_,
        config_name: Optional[str] = None,
):
    """
    Decorate register_pipelines()

    Args:
        config_path: The config path, a directory relative to the declaring python file.
                     If config_path is None no directory is added to the Config search path.
        config_name: The name of the config (usually the file name without the .yaml extension)

    """

    # save sys.argv state before adapting it to the hydra decorator
    bckp_argv = deepcopy(sys.argv)

    # initialise sys.argv with only the running script path (../env/ENV_NAME/bin/kedro)
    sys.argv = [sys.argv[0]]

    # fetch the extra parameters given to kedro via the --params flag
    try:
        kedro_session = get_current_session()
    except:
        kedro_session = None
    if kedro_session:
        kedro_context = kedro_session.load_context()
        kedro_extra_params = kedro_context.params

        if 'hydra' in kedro_extra_params:
            hydra_overrides = shlex.split(kedro_extra_params['hydra'])
            sys.argv = sys.argv + hydra_overrides

    def root_decorator(task_fun: TaskFunction):
        global __name__
        bckp__name__ = deepcopy(__name__)
        __name__ = '__main__'

        taskfun_filepath = inspect.getfile(task_fun)
        taskfun_path = os.path.split(taskfun_filepath)[0]

        decorator_filepath = os.path.realpath(__file__)
        decorator_folder = os.path.split(decorator_filepath)[0]

        config_realpath = os.path.realpath(os.path.join(taskfun_path, config_path))
        config_relpath = os.path.relpath(path=config_realpath, start=decorator_folder)

        @hydra.main(config_relpath, config_name)
        def task_function_execution(cfg: DictConfig):
            # restore sys.argv state before running hydra decorator
            sys.argv = bckp_argv
            global __name__
            __name__ = bckp__name__
            # load the cfg and its branches into this session's catalog
            update_catalog_extension(cfg)

            # run the task function, register_pipelines() for example
            # then update the global variable _pipelines_registered to allow for a return outside of hydra
            global _pipelines_registered
            _pipelines_registered = task_fun(cfg)

        @functools.wraps(task_fun)
        def bridge_fun(*args, **kwargs):
            """
            Bridge the output of the function decorated by hydra.main
            to the caller of said function.
            """
            # run the decorated pipeline_registry_fun
            task_function_execution()  # updates the global variable _pipelines_registered

            # then return to the caller the output of the function decorated with hydra.main
            global _pipelines_registered
            return _pipelines_registered

        return bridge_fun

    return root_decorator


