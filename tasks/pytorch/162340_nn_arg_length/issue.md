# nn module rejects iterable arguments of incorrect length

## Bug Description

torch.nn.modules.conv and pooling modules accept tuple/list arguments for kernel_size, stride, padding etc. but do not validate that the iterable length matches the expected dimension count, leading to silent incorrect behavior or confusing errors downstream.

## Reference

- Issue: https://github.com/pytorch/pytorch/issues/162327
- Fix PR: https://github.com/pytorch/pytorch/pull/162340
