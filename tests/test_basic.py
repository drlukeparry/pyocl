# -*- coding: utf-8 -*-
from .context import pyocl

import unittest
import platform
import tempfile


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_absolute_truth_and_meaning(self):
        assert True


if __name__ == '__main__':
    unittest.main()