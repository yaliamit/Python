
import numpy as np

def zeropad(array, pad_width):
    shape = list(array.shape)
    slices = []
    shape = [array.shape[i] + pad_width[i] * 2 for i in range(len(array.shape))]
    slices = [slice(pad_width[i], pad_width[i]+array.shape[i]) for i in range(len(array.shape))]
    new_array = np.zeros(shape)
    new_array[slices] = array
    return new_array
