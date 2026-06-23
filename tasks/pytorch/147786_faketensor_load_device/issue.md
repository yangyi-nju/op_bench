# torch.load under FakeTensorMode does not preserve original device for plain Tensors

## Bug Description

When `torch.load` is called inside a `FakeTensorMode` context, plain (non-storage-attached) tensors lose their original device (e.g., loaded as CPU when the saved tensor was CUDA). The fix preserves the device metadata through serialization unpickling so FakeTensors carry the correct device.

## Reference

- PR: https://github.com/pytorch/pytorch/pull/147786
