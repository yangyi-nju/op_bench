# `LazyLinear` keeps `in_features=0` after loading initialized state and running forward

Issue: https://github.com/pytorch/pytorch/issues/147389
PR: https://github.com/pytorch/pytorch/pull/147599

`nn.LazyLinear` can load an initialized `nn.Linear` state dict before its first forward pass. In the buggy behavior, the parameters are already materialized, so `initialize_parameters()` skips its lazy branch and never updates `in_features`; after a forward pass the module becomes `nn.Linear` but still reports `in_features=0`.

The upstream fix updates `in_features` from the loaded weight shape when initialized state has already been loaded.
