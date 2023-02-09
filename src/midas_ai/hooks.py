import ray
from kedro.framework.hooks import hook_impl
from ray.util.dask import enable_dask_on_ray


class ProjectHooks:
    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog):
        ray.init(address="auto")
        enable_dask_on_ray()
