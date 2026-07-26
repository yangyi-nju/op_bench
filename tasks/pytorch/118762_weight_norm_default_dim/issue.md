# Weight norm decomposition omits the native default dim

The native `_weight_norm_interface` schema accepts two arguments and defaults
`dim` to zero. The Python decomposition requires `dim`, so the same call fails
when decomposition dispatch is selected. Match the native signature without
changing explicit-dimension behavior.

Only `torch/_decomp/decompositions.py` may be modified.
