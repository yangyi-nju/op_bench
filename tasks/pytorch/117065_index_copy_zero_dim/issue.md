# index_copy decomposition rejects a zero-dimensional index

Native `index_copy` accepts a scalar index as a one-element index. The Python
decomposition validates that rank but passes it unchanged to `index_put`,
causing the decomposition path to fail. Normalize only the scalar case and
preserve the existing one-dimensional behavior.

Only `torch/_decomp/decompositions.py` may be modified.
