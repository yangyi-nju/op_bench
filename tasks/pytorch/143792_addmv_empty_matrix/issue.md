# Inductor addmv decomposition mishandles a zero-sized matrix

`torch.addmv` accepts a broadcastable input even when the matrix is `0 x 0`.
The decomposition's generic `out + beta * self` path does not preserve the
native empty-matrix result. Fix the empty-result branch without changing the
existing `beta=0` behavior.

Only `torch/_decomp/decompositions.py` may be modified.
