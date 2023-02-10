from flatten_dict import flatten
from kedro.framework.hooks.manager import get_hook_manager
from kedro.io import DataCatalog
from ts_pattern_miner.hooks import ProjectHooks
from omegaconf import DictConfig

# A function decorated with hydra.main can only return None.
# This global variable is used to get a return anyway when a function
#  is decorated with the decorator in this script.
_pipelines_registered = None


def update_catalog_extension(cfg: DictConfig):
    """
    The catalog is recreated at each call of context.catalog, which happens later in kedro's code,
     before running a pipeline for example.
    This function will modify the registered ProjectHooks to set him an attribute
     containing the hydra config in catalog format.
    The hook ProjectHooks has been modified to add this extension to the session's catalog
     everytime it is called to compute it.

    Args:
        cfg:

    Returns:

    """

    # Creates a datacatalog containing the hydra config
    catalog = DataCatalog()
    catalog.add_feed_dict({'config': cfg}, replace=True)
    max_cfg_depth = max(map(len, flatten(cfg).keys()))
    feed_dict = {}
    for depth in range(1, max_cfg_depth):
        for keys_to_parameter, parameter in flatten(cfg, max_flatten_depth=depth).items():
            feed_dict.update({f'params:{".".join(keys_to_parameter)}': parameter})
            # if keys_to_parameter[0] == 'parameters':
            #     feed_dict.update({f'params:{">".join(keys_to_parameter[1:])}': parameter})
            # else:
            #     feed_dict.update({f'cfg:{">".join(keys_to_parameter)}': parameter})
    catalog.add_feed_dict(feed_dict, replace=True)

    # updates the datacalog_extension attribute of the registered ProjectHook
    hook_manager = get_hook_manager()
    for k, v in hook_manager._plugin2hookcallers.items():
        if isinstance(k, ProjectHooks):
            k.__setattr__('datacatalog_extension', catalog)
