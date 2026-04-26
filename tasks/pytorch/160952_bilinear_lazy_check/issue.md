# `Bilinear` module cannot be made lazy due to new checks in `__init__`

Issue: https://github.com/pytorch/pytorch/issues/160407
PR: https://github.com/pytorch/pytorch/pull/160952

`nn.Bilinear` rejects `in1_features <= 0` inside `__init__`, which blocks implementing a lazy variant that starts with unknown input sizes. The proposed upstream fix moves the check closer to the division in `reset_parameters()`.

This task is a candidate and still requires a replayed hidden regression test before being marked verified.
