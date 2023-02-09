from typing import Union, Iterable, Set, Dict

from kedro.pipeline import Pipeline, pipeline
from kedro.pipeline.node import Node


class ParametricPipeline(Pipeline):

    def __init__(self, pipe: Union[Iterable[Union[Node, Pipeline]], Pipeline]):
        """

        Args:
            pipe:
        """
        if not isinstance(pipe, Iterable):
            pipe = [pipe]
        super().__init__(pipe, tags=None)

    @classmethod
    def create_pipeline_vanilla(
            cls,
            pipe: Union[Iterable[Union[Node, Pipeline]], Pipeline],
            *,
            inputs: Union[str, Set[str], Dict[str, str]] = None,
            outputs: Union[str, Set[str], Dict[str, str]] = None,
            parameters: Dict[str, str] = None,
            namespace: str = None
    ) -> Pipeline:
        return pipeline(
            cls(pipe),
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            namespace=namespace
        )

    @classmethod
    def create_pipeline_from_tags(
            cls,
            pipe: Union[Iterable[Union[Node, Pipeline]], Pipeline],
            use_tags: Union[str, Iterable[str]] = None,
            *,
            inputs: Union[str, Set[str], Dict[str, str]] = None,
            outputs: Union[str, Set[str], Dict[str, str]] = None,
            parameters: Dict[str, str] = None,
            namespace: str = None
    ) -> Pipeline:
        if not isinstance(use_tags, Iterable):
            use_tags = [use_tags]

        return pipeline(
            cls(pipe),
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            namespace=namespace
        ).only_nodes_with_tags(*use_tags)

    @classmethod
    def create_pipeline_from_nodes(
            cls,
            pipe: Union[Iterable[Union[Node, Pipeline]], Pipeline],
            use_nodes: Union[str, Iterable[str]] = None,
            *,
            inputs: Union[str, Set[str], Dict[str, str]] = None,
            outputs: Union[str, Set[str], Dict[str, str]] = None,
            parameters: Dict[str, str] = None,
            namespace: str = None
    ) -> Pipeline:
        if not isinstance(use_nodes, Iterable):
            use_nodes = [use_nodes]

        return pipeline(
            cls(pipe),
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            namespace=namespace
        ).only_nodes(*use_nodes)

    @staticmethod
    def check_io_type_compatibility(prior_pipeline: Pipeline, posterior_pipeline: Pipeline):
        raise NotImplementedError
