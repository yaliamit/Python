from __future__ import division, print_function, absolute_import
import amitgroup as ag
import unittest


class TestIO(unittest.TestCase):
    def setUp(self):
        pass

    def test_example(self):
        data = ag.io.load_example('two-faces')
        self.assertEqual(data.shape, (2, 32, 32))

        data2 = ag.io.load_example('mnist')
        self.assertEqual(data2.shape, (10, 28, 28))


if __name__ == '__main__':
    unittest.main()
