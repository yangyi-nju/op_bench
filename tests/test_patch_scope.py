"""Tests for op_bench.patch_scope module."""

import unittest

from op_bench.patch_scope import PatchScopeResult, extract_patch_paths, validate_patch_scope


SAMPLE_PATCH = """\
diff --git a/torch/nn/modules/linear.py b/torch/nn/modules/linear.py
index abc1234..def5678 100644
--- a/torch/nn/modules/linear.py
+++ b/torch/nn/modules/linear.py
@@ -10,6 +10,7 @@ class Linear(Module):
     def __init__(self):
         super().__init__()
+        self.reset_parameters()

diff --git a/torch/nn/modules/lazy.py b/torch/nn/modules/lazy.py
index 111aaaa..222bbbb 100644
--- a/torch/nn/modules/lazy.py
+++ b/torch/nn/modules/lazy.py
@@ -5,3 +5,4 @@ class LazyModuleMixin:
     pass
+    # fix
"""

SINGLE_FILE_PATCH = """\
diff --git a/torch/nn/modules/linear.py b/torch/nn/modules/linear.py
index abc1234..def5678 100644
--- a/torch/nn/modules/linear.py
+++ b/torch/nn/modules/linear.py
@@ -10,6 +10,7 @@ class Linear(Module):
     def __init__(self):
         super().__init__()
+        self.reset_parameters()
"""


class TestExtractPatchPaths(unittest.TestCase):
    def test_multi_file(self):
        paths = extract_patch_paths(SAMPLE_PATCH)
        self.assertEqual(paths, ["torch/nn/modules/linear.py", "torch/nn/modules/lazy.py"])

    def test_single_file(self):
        paths = extract_patch_paths(SINGLE_FILE_PATCH)
        self.assertEqual(paths, ["torch/nn/modules/linear.py"])

    def test_empty(self):
        self.assertEqual(extract_patch_paths(""), [])


class TestValidatePatchScope(unittest.TestCase):
    def test_no_scope_returns_no_scope(self):
        result = validate_patch_scope(SAMPLE_PATCH, [], "enforced")
        self.assertEqual(result.status, "no_scope")
        self.assertEqual(result.filtered_patch, SAMPLE_PATCH)

    def test_empty_patch(self):
        result = validate_patch_scope("", ["torch/nn/modules/linear.py"], "enforced")
        self.assertEqual(result.status, "empty_patch")

    def test_in_scope(self):
        allowed = ["torch/nn/modules/linear.py", "torch/nn/modules/lazy.py"]
        result = validate_patch_scope(SAMPLE_PATCH, allowed, "enforced")
        self.assertEqual(result.status, "in_scope")
        self.assertEqual(result.out_of_scope_paths, [])

    def test_enforced_out_of_scope(self):
        allowed = ["torch/nn/modules/linear.py"]
        result = validate_patch_scope(SAMPLE_PATCH, allowed, "enforced")
        self.assertEqual(result.status, "out_of_scope")
        self.assertEqual(result.out_of_scope_paths, ["torch/nn/modules/lazy.py"])
        self.assertEqual(result.filtered_patch, "")

    def test_filtered_mode(self):
        allowed = ["torch/nn/modules/linear.py"]
        result = validate_patch_scope(SAMPLE_PATCH, allowed, "filtered")
        self.assertEqual(result.status, "filtered")
        self.assertEqual(result.out_of_scope_paths, ["torch/nn/modules/lazy.py"])
        self.assertIn("torch/nn/modules/linear.py", result.filtered_patch)
        self.assertNotIn("torch/nn/modules/lazy.py", result.filtered_patch)

    def test_single_file_in_scope(self):
        allowed = ["torch/nn/modules/linear.py"]
        result = validate_patch_scope(SINGLE_FILE_PATCH, allowed, "enforced")
        self.assertEqual(result.status, "in_scope")


if __name__ == "__main__":
    unittest.main()
