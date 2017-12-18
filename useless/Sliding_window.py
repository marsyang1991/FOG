import numpy as np
from numpy.lib.stride_tricks import as_strided as ast


def norm_shape(shape):
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        pass
    try:
        t = tuple(shape)
        return t
    except TypeError:
        pass
    raise TypeError('shape must be an int or tuple of ints')


def sliding_window(a, ws, ss=None, flatten=True):
    if None is ss:  # ss is not provided, the windows will not overlap
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)
    newshape = norm_shape(((shape - ws) // ss) + 1)
    newshape += norm_shape(ws)
    newstrides = norm_shape(np.array(a.strides * ss) + a.strides)
    strided = ast(a, shape=newshape, strides=newstrides)
    if not flatten:
        return strided
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:,-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    dim = filter(lambda i : i != 1, dim)
    return strided.reshape(dim)