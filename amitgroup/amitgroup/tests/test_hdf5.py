from __future__ import division, print_function, absolute_import
import amitgroup as ag
import unittest
import numpy as np


class TestHDF5(unittest.TestCase):
    def setUp(self):
        pass

    def _reconstruct(self, x):
        import tempfile
        fn = tempfile.NamedTemporaryFile(delete=True).name
        ag.io.save(fn, x)
        return ag.io.load(fn)

    def test_hdf5_basic(self):
        subdict = {'test': 200}
        x = dict(a=10,
                 b=12.31,
                 c=[12, [1, subdict], 14],
                 d=(1.2, 20),
                 e=subdict,
                 f=None)
        y = self._reconstruct(x)
        assert x == y, (x, y)

    def test_hdf5_int_key(self):
        x = {100: [12, 13]}
        y = self._reconstruct(x)
        assert x == y, (x, y)

    def test_hdf5_tuple_key(self):
        x = {(100, 200): 200}
        y = self._reconstruct(x)
        assert x == y, (x, y)

    def test_hdf5_string(self):
        x = u'this is a string'
        y = self._reconstruct(x)
        assert x == y, (x, y)

    def test_hdf5_numpy_array(self):
        x = np.arange(20)
        y = self._reconstruct(x)
        np.testing.assert_array_equal(x, y)

    def test_hdf5_numpy_array_inside(self):
        x = dict(a=np.arange(10),
                 b=np.ones(1, dtype=np.bool_),
                 c=[np.ones(10), np.array([1.23, 1.34], dtype=np.float32)])
        y = self._reconstruct(x)
        assert isinstance(y, dict)
        assert set(y.keys()) == set(x.keys())
        np.testing.assert_array_equal(x['a'], y['a'])
        np.testing.assert_array_equal(x['b'], y['b'])
        assert isinstance(y['c'], list)
        np.testing.assert_array_equal(x['c'][0], y['c'][0])
        np.testing.assert_array_equal(x['c'][1], y['c'][1])


if __name__ == '__main__':
    unittest.main()
