# `torch.special.expit` should preserve NaN values on CPU

On CPU builds, `torch.special.expit` should preserve NaN inputs. The current operator path returns a finite value for NaN, which masks invalid numerical inputs and diverges from the documented behavior.
