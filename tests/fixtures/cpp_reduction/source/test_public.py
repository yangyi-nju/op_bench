"""Public reproduction: compile operator.cpp, then load that workspace's library."""

import ctypes
import math
from pathlib import Path
import sys
import unittest


class PublicRowSumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        library = Path(__file__).resolve().parent / "liboperator.so"
        cls.library = ctypes.CDLL(str(library))
        cls.row_sum = cls.library.row_sum
        cls.row_sum.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                               ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        cls.row_sum.restype = None

    def reduce(self, matrix):
        rows, cols = len(matrix), len(matrix[0])
        data = (ctypes.c_double * (rows * cols))(*(value for row in matrix for value in row))
        output = (ctypes.c_double * rows)()
        self.row_sum(data, rows, cols, output)
        return list(output)

    def test_distinct_rows(self):
        self.assertEqual(self.reduce([[1.0, 2.0], [4.0, 5.0]]), [3.0, 9.0])

    def test_single_row(self):
        self.assertEqual(self.reduce([[2.0, -1.0, 4.0]]), [5.0])

    def test_cancellation_with_rounding_budget(self):
        values = [1_000_000.0, 0.1, -1_000_000.0]
        expected = math.fsum(values)
        budget = 1e-12 + 2 * len(values) * sys.float_info.epsilon * math.fsum(map(abs, values))
        self.assertLessEqual(abs(self.reduce([values])[0] - expected), budget)


if __name__ == "__main__":
    unittest.main()
