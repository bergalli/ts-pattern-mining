from typing import List

from dask import array as da


def _stack_vectors_columns(dask_arrays: List[da.Array], dask_chunk_size="100Mb") -> da.Array:
    stacked_vectors = da.vstack(dask_arrays)
    stacked_vectors = stacked_vectors.T
    stacked_vectors = stacked_vectors.rechunk(chunks=(dask_chunk_size, stacked_vectors.shape[1]))
    return stacked_vectors