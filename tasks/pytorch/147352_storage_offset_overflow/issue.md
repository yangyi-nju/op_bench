# as_strided storage-offset bounds check can overflow

`checkInBoundsForStorage` computes `storage_offset * itemsize` and adds the
required storage size without checked arithmetic. A very large `int64` offset
can wrap and pass the bounds check. Use the existing checked storage-size
helper with the offset included.

The early return for zero-size tensors is intentional and must remain valid.
Only `aten/src/ATen/native/Resize.h` may be modified.
