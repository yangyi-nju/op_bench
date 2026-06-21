# lr_scheduler and swa_utils fail to deepcopy lr and base parameters

## Bug Description

When using deepcopy on lr_scheduler objects, the lr values and other base_parameters are shared between the original and the copy. Modifying lr in one scheduler unexpectedly affects the other.

## Reference

- Issue: https://github.com/pytorch/pytorch/issues/126854
- Fix PR: https://github.com/pytorch/pytorch/pull/127190
