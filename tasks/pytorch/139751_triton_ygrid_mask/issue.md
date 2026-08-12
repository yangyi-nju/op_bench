# Triton y-grid overflow can invalidate a constant mask

`TritonKernel._has_constant_mask` may remove a mask whenever `numel` is a
multiple of the block size. For a logical Y dimension beyond the hardware
y-grid limit, wrapping through Z can still expose out-of-range lanes. Retain
the mask for an overflowing Y grid unless the range tree already has a Z
dimension.

The hidden test is a low-memory surrogate of the production predicate and does
not allocate a grid-sized tensor. Only
`torch/_inductor/codegen/triton.py` may be modified.
