"""Independent behavior checks against the library built in the evaluated workspace."""

import ctypes
import math
from pathlib import Path
import sys
import unittest


class RowSumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        library = (workspace / "liboperator.so").resolve()
        if not library.is_relative_to(workspace) or not library.is_file():
            raise RuntimeError("liboperator.so must be built inside the evaluated workspace")
        cls.library = ctypes.CDLL(str(library))
        cls.row_sum = cls.library.row_sum
        cls.row_sum.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                               ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        cls.row_sum.restype = None

    def assert_rows(self, matrix):
        rows, cols = len(matrix), len(matrix[0])
        self.assertTrue(all(len(row) == cols for row in matrix))
        flat = [value for row in matrix for value in row]
        data = (ctypes.c_double * len(flat))(*flat)
        before = list(data)
        sentinel = 1234567.25
        guarded_output = (ctypes.c_double * (rows + 2))(*([sentinel] * (rows + 2)))
        output = ctypes.cast(ctypes.byref(guarded_output, ctypes.sizeof(ctypes.c_double)),
                             ctypes.POINTER(ctypes.c_double))
        self.row_sum(data, rows, cols, output)
        self.assertEqual(list(data), before, "row_sum must preserve the input array")
        self.assertEqual(guarded_output[0], sentinel, "write before output buffer")
        self.assertEqual(guarded_output[-1], sentinel, "write after output buffer")
        for row, expected in enumerate(map(math.fsum, matrix)):
            self.assertTrue(math.isfinite(output[row]))
            magnitude = math.fsum(abs(value) for value in matrix[row])
            error_budget = 1e-12 + 2 * cols * sys.float_info.epsilon * magnitude
            self.assertLessEqual(abs(output[row] - expected), error_budget,
                                 f"row {row}: expected {expected}, received {output[row]}, budget {error_budget}")

    def test_distinct_rows(self):
        self.assert_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])

    def test_rectangular_shapes(self):
        for rows, cols in ((2, 7), (5, 2), (4, 1)):
            with self.subTest(rows=rows, cols=cols):
                self.assert_rows([[float((row + 1) * (col + 2)) for col in range(cols)] for row in range(rows)])

    def test_negative_and_zero_values(self):
        self.assert_rows([[0.0, -2.0, 0.0, -3.0], [-1.0, 0.0, 0.0, -7.0],
                          [0.0, 0.0, 0.0, 0.0], [1.5, -0.5, 0.0, 2.25]])

    def test_single_row(self):
        self.assert_rows([[2.0, -1.0, 4.0, 0.5]])

    def test_single_element(self):
        self.assert_rows([[-2.5]])

    def test_zero_single_row(self):
        self.assert_rows([[0.0, 0.0, 0.0]])

    def test_cancellation(self):
        for values in ([1_000_000.0, 0.1, -1_000_000.0],
                       [1_000_000.0, *([0.1] * 1000), -1_000_000.0]):
            with self.subTest(cols=len(values)):
                self.assert_rows([values])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: test_reduction.py WORKSPACE [RowSumTests.test_name] [-v]")
    workspace = Path(sys.argv.pop(1)).resolve()
    unittest.main()
