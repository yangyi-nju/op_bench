# load_state_dict silently accepts unexpected keys whose prefix matches a valid key

## Bug Description

When loading a state_dict with strict=True, unexpected keys whose name starts with a prefix that matches a valid parameter key are incorrectly treated as valid, allowing corrupt state dicts to load without error.

## Reference

- Issue: https://github.com/pytorch/pytorch/issues/123510
- Fix PR: https://github.com/pytorch/pytorch/pull/124385
