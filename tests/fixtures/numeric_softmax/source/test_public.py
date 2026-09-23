"""Public behavior feedback; does not import a private oracle or grader."""
import math
import unittest

from ops import softmax


class PublicSoftmax(unittest.TestCase):
    def test_shifted_pair(self):
        result = softmax([999.0, 1000.0])
        self.assertEqual(len(result), 2)
        expected = [1 / (1 + math.e), math.e / (1 + math.e)]
        for actual, reference in zip(result, expected):
            self.assertAlmostEqual(actual, reference, places=12)

    def test_equal_pair(self):
        self.assertEqual(softmax([0.0, 0.0]), [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
