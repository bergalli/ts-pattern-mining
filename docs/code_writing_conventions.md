# Kedro

## Elementary/Fundamental pipelines

Made of nodes

- node params:
    - inputs : always represented as a dictionary
    - outputs : single as string, multiple as list. Suffixed with `'_a'`, `'_b'`, .. if reuse of same node. Explicity
      None if output is None
    - name : fun name suffixed with `'_node'` or `'node_a'`, `'node_b'`, .. if reuse of same node

- pipeline params:
    - inputs always as a set of strings. Defined are parameters not available in `cfg:...`. Names corresponds to the
      dictionary values of the inputs.
    - outputs as a set or dictionary. Dictionary in case output has little meaning, then mapping from names in nodes
      outputs, to names coherent with pipeline's name.

In file `<package>/pipelines/pipeline_1/sub_pipeline_1/pipeline.py` :

```python
from .nodes import fun1, fun2


def create_pipeline(**kwargs) -> Pipeline:
  return pipeline(
    [
      node(
        fun1,  # 3 inputs 1 output
        inputs=dict(param1="param1",  # e.g. dataset of datacatalog
                    param2="param2",
                    param3="cfg:sub_pipeline_1>param3"),
        outputs=["output_1", "output_2"],
        name="<fun1>_node"
      ),
      node(
        fun2,  # 1 input 2 outputs
        inputs=dict(param1="output_1"),
        outputs="output3_a",
        name="<fun2>_node_a"
      ),
      node(
        fun2,  # 1 input 2 outputs
        inputs=dict(param1="output_2"),
        outputs="output3_b",
        name="<fun2>_node_b"
      )
    ],
    inputs={"param1"},
    outputs={"output3_a", "output3_b"} or {"output3_a": "<sub_pipeline_1>_output_3_a",
                                           "output3_b": "<sub_pipeline_1>_output_3_b"}
  )
```

## Macro/Compound/Aggregate/Composite/Secondary pipelines

Made of pipelines

In file `<package>/pipelines/pipeline_1/nodes.py`

```python

```

In file `<package>/pipelines/pipeline_1/pipeline.py`

```python

```

## Mixed/Hybrid pipelines

Made of nodes and pipelines

# Hydra

fonctions dans pipeline prennent cfg en param fonctions dans steps prennent config_groups en param fonctions dans
dossiers de fonctionnement prennent tous params
