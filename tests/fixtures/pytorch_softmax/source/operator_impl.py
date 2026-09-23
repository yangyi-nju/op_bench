"""Synthetic operator implementation for the OpBench integration example."""

import torch


def stable_softmax(input, dim=-1):
    """Return softmax probabilities along dim for a finite floating tensor."""
    exponentials = torch.exp(input)
    return exponentials / exponentials.sum(dim=dim, keepdim=True)
