# nn.Module.set_submodule fails for non dot-delineated target strings

## Bug Description

nn.Module.set_submodule does not correctly handle target strings without dot separators, causing it to silently create nested modules with wrong names or fail to set the submodule at the expected path.

## Reference

- Issue: https://github.com/pytorch/pytorch/issues/143441
- Fix PR: https://github.com/pytorch/pytorch/pull/143455
