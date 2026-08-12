# Inductor cummin and cummax lowerings fail for rank-zero inputs

The scalar lowering returns a cloned value plus an `int64` index. It currently
uses eager `torch.empty_like` on an Inductor IR value, which cannot be lowered.
Use the lowering helper for the index output while preserving the normal scan
path for rank-one and larger inputs.

Only `torch/_inductor/lowering.py` may be modified.
