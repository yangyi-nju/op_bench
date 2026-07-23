import unittest

from calc import normalize


class PublicTests(unittest.TestCase):
    def test_number_is_preserved(self):
        self.assertEqual(normalize(1), 1)
