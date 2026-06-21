# Unused gradient tracking does not respect create_graph flag

## Bug Description

When calling torch.autograd.grad with create_graph=True and some inputs have unused gradients, the unused gradient tracking incorrectly handles the create_graph flag, leading to incorrect higher-order gradient computation.

## Reference

- Issue: https://github.com/pytorch/pytorch/issues/168059
- Fix PR: https://github.com/pytorch/pytorch/pull/168295
