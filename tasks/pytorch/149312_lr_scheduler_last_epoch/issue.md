# lr_scheduler unexpectedly calls step() when last_epoch > -1

## Bug Description

When initializing a learning rate scheduler with last_epoch greater than -1 (e.g., for checkpoint resumption), the scheduler incorrectly calls step() during __init__, causing the learning rate to advance one step beyond expected.

## Reference

- Issue: https://github.com/pytorch/pytorch/issues/102261
- Fix PR: https://github.com/pytorch/pytorch/pull/149312
