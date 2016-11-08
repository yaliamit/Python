from __future__ import division, print_function, absolute_import
import numpy as np
import unittest
import os
import amitgroup.util.wavelet as wv


def rel(x):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), x)


class TestWavelet(unittest.TestCase):
    def setUp(self):
        pass

    def _test_wavedec(self, wavelet, length, levels):
        import pywt
        ll = int(np.log2(length))
        A = np.arange(length)

        coefs = pywt.wavedec(A, wavelet, mode='per', level=ll)
        # including 0th level as one
        u_ref = wv.structured_to_contiguous(coefs[:levels+1])

        wavedec, waverec = wv.daubechies_factory(length, wavelet=wavelet)
        u = wavedec(A, levels=levels)

        np.testing.assert_array_almost_equal(u_ref, u)
        return u

    def test_wavedec_8(self):
        for i in range(0, 4):
            self._test_wavedec('db2', 8, i)

    def test_wavedec_16(self):
        for i in range(0, 5):
            self._test_wavedec('db3', 16, i)

    def test_wavedec_32(self):
        for i in range(0, 6):
            self._test_wavedec('db4', 32, i)

    def _test_waverec(self, wavelet, length, levels):
        A = np.arange(length)
        ll = int(np.log2(length))

        N = 1 << levels
        # This assumes wavedec is working
        u = self._test_wavedec(wavelet, length, levels)

        u_zeros = np.zeros(len(u))
        u_zeros[:N] = u[:N]

        # Reconstruction
        wavedec, waverec = wv.daubechies_factory(length, wavelet=wavelet)

        A_rec_ref = waverec(u_zeros)
        A_rec = waverec(u)

        if levels == ll:
            np.testing.assert_array_almost_equal(A, A_rec)
        else:
            # They should not be equal, since the image will have lost
            # integrity
            assert not (A == A_rec).all()
            np.testing.assert_array_almost_equal(A_rec_ref, A_rec)

    def test_waverec_8(self):
        for i in range(0, 4):
            self._test_waverec('db2', 8, i)

    def test_waverec_16(self):
        for i in range(0, 5):
            self._test_waverec('db3', 16, i)

    def test_waverec_32(self):
        for i in range(0, 6):
            self._test_waverec('db4', 32, i)

    # Test all wavelet types.
    def test_wavedec_all_daubechies(self):
        for i in range(1, 21):
            self._test_wavedec('db{0}'.format(i), 8, i)

    def test_waverec_all_daubechies(self):
        for i in range(1, 21):
            self._test_waverec('db{0}'.format(i), 8, i)

    # 2-D

    def _test_wavedec2(self, wavelet, shape, levels):
        import pywt
        ll = int(np.log2(max(shape)))
        A = np.arange(np.prod(shape)).reshape(shape)

        coefs = pywt.wavedec2(A, wavelet, mode='per', level=ll)
        # including 0th level as one
        u_ref = wv.structured_to_contiguous(coefs[:levels+1])

        wavedec2, waverec2 = wv.daubechies_factory(shape, wavelet=wavelet)
        u = wavedec2(A, levels=levels)

        np.testing.assert_array_almost_equal(u_ref, u)
        return u

    def test_wavedec2_16(self):
        for i in range(0, 5):
            self._test_wavedec2('db2', (16, 16), i)

    def test_wavedec2_32(self):
        for i in range(0, 6):
            self._test_wavedec2('db2', (32, 32), i)

    def test_wavedec2_64(self):
        for i in range(0, 7):
            self._test_wavedec2('db2', (64, 64), i)

    def _test_waverec2(self, wavelet, shape, levels):
        A = np.arange(np.prod(shape)).reshape(shape)
        ll = int(np.log2(max(shape)))

        N = 1 << levels
        # This assumes wavedec2 is working
        u = self._test_wavedec2(wavelet, shape, levels)

        u_zeros = np.zeros(u.shape)
        u_zeros[:N, :N] = u[:N, :N]

        # Reconstruction
        wavedec2, waverec2 = wv.daubechies_factory(A.shape, wavelet=wavelet)

        A_rec_ref = waverec2(u_zeros)
        A_rec = waverec2(u)

        if levels == ll:
            np.testing.assert_array_almost_equal(A, A_rec)
        else:
            # They should not be equal, since the image will have lost
            # integrity
            assert not (A == A_rec).all()
            np.testing.assert_array_almost_equal(A_rec_ref, A_rec)

    def test_waverec2_16(self):
        for i in range(0, 5):
            self._test_waverec2('db2', (16, 16), i)

    def test_waverec2_32(self):
        for i in range(0, 6):
            self._test_waverec2('db2', (32, 32), i)

    def test_waverec2_64(self):
        for i in range(0, 7):
            self._test_waverec2('db2', (64, 64), i)

    # Non-square
    @unittest.skip("Not implemented yet.")
    def test_wavedec2_32_16(self):
        for i in range(0, 5):
            self._test_wavedec2('db2', (32, 16), i)

    @unittest.skip("Not implemented yet.")
    def test_wavedec2_16_32(self):
        for i in range(0, 5):
            self._test_wavedec2('db2', (16, 32), i)

    # Test all wavelet types.
    def test_wavedec2_all_daubechies(self):
        for i in range(1, 21):
            self._test_wavedec2('db{0}'.format(i), (8, 8), i)

    def test_waverec2_all_daubechies(self):
        for i in range(1, 21):
            self._test_waverec2('db{0}'.format(i), (8, 8), i)

    def _test_cache(self, shape, wavelet, levels):
        A = np.arange(np.prod(shape)).reshape(shape)
        wavedec2, waverec2 = wv.daubechies_factory(shape, wavelet)
        np.testing.assert_array_equal(wv.wavedec2(A, wavelet,
                                      levels), wavedec2(A, levels))
        np.testing.assert_array_equal(wv.waverec2(A, wavelet),
                                      waverec2(A))

    def test_wavelet_cache(self):
        for shape in [(8, 8), (16, 16), (8, 8)]:
            self._test_cache(shape, 'db4', 2)


if __name__ == '__main__':
    unittest.main()
