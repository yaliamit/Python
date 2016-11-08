from __future__ import division, print_function, absolute_import 
import amitgroup as ag
import unittest
import numpy as np


class TestUtil(unittest.TestCase):
    def setUp(self):
        pass

    def test_border_value_pad(self):
        x = np.arange(8).reshape((2, 2, 2))
        output = ag.util.pad_repeat_border(x, (2, 0, 1))
        correct = np.array([[[0, 0, 1, 1], [2, 2, 3, 3]]] * 3 +
                            [[[4, 4, 5, 5], [6, 6, 7, 7]]] * 3)

        np.testing.assert_array_equal(output, correct)

    def test_inflate2d_single(self):
        x = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ], dtype=np.uint8)

        # Square kernel
        correct_y = np.array([
            [1, 1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ], dtype=np.uint8)

        # Diagonal kernel
        correct_z = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
        ], dtype=np.uint8)

        y = ag.util.inflate2d(x, np.ones((3, 3), dtype=np.uint8))
        np.testing.assert_array_equal(y, correct_y)
        z = ag.util.inflate2d(x, np.eye(3, dtype=np.uint8))
        np.testing.assert_array_equal(z, correct_z)

        # Test with several levels
        x2 = x.reshape((1, 1, 1, 1) + x.shape)
        correct_y2 = y.reshape((1, 1, 1, 1) + y.shape)
        correct_z2 = z.reshape((1, 1, 1, 1) + z.shape)
        y2 = ag.util.inflate2d(x2, np.ones((3, 3), dtype=np.uint8))
        np.testing.assert_array_equal(y2, correct_y2)
        z2 = ag.util.inflate2d(x2, np.eye(3, dtype=np.uint8))
        np.testing.assert_array_equal(z2, correct_z2)


if __name__ == '__main__':
    unittest.main()
