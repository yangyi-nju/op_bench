# LBFGS optimizer wolfe line search max iteration is incorrect

## Bug Description

The LBFGS optimizer strong_wolfe line search uses an incorrect max iteration count, causing it to terminate too early and produce suboptimal step sizes.

## Reference

- Issue: https://github.com/pytorch/pytorch/issues/91581
- Fix PR: https://github.com/pytorch/pytorch/pull/161488
