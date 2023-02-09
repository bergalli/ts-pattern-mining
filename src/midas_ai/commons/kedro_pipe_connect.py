from kedro.pipeline import Pipeline, pipeline
from typing import Dict


def pipe_connect_out_to_in(
    from_pipe: Pipeline, to_pipe: Pipeline, from_out_to_in: Dict[str, str] = None
) -> Pipeline:

    ref_out = from_pipe.outputs()
    ref_in = to_pipe.inputs()
    assert len(ref_out) == len(ref_in)

    oi_mapper = {
        **{},
        **{from_o: from_o for from_o in ref_out if from_o in ref_in},
        **{
            from_o: from_out_newname
            for from_o, from_out_newname in from_out_to_in.items()
            if from_o in ref_in
        },
    }

    return pipeline(from_pipe, outputs=oi_mapper) + to_pipe
