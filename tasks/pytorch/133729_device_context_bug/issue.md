# DeviceContext does not properly track default device under nested usage

## Bug Description

The `torch.utils._device.DeviceContext` (used by `torch.set_default_device`) has a state tracking bug that causes incorrect device propagation when nested or used with Dynamo. The fix corrects the context state machine to properly restore the previous default device.

## Reference

- PR: https://github.com/pytorch/pytorch/pull/133729
