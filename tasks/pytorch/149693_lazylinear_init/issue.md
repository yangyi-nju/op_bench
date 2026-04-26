# LazyLinear broken by new init logic

Issue: https://github.com/pytorch/pytorch/issues/149691
PR: https://github.com/pytorch/pytorch/pull/149693

`nn.LazyLinear` initializes with zeroed output after a recent initialization logic change. The issue reproducer shows that calling a `LazyLinear` layer returns zeros because `in_features` remains zero when `reset_parameters()` is called during lazy initialization.

Expected behavior: lazy initialization should materialize `in_features` from the input shape before resetting parameters, so initialized weights, bias, and output are non-zero for normal random initialization.
