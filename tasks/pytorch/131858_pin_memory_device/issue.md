# pin_memory APIs incorrectly pass device argument breaking custom backends

## Bug Description

Several pin-memory related APIs (Tensor.pin_memory, Storage.pin_memory, DataLoader pin_memory) pass a `device` argument that does not respect the privateuse1 / custom backend conventions. The fix updates these APIs to not pass `device` explicitly, deferring to the default backend.

## Reference

- PR: https://github.com/pytorch/pytorch/pull/131858
