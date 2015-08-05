import numpy as np

def _vectorize(arr):
    h,w = arr.shape
    num = h*w
    return np.reshape(arr, (num))

def pix2vec(im):
    pix = im.load()
    w,h = im.size
    arr = np.zeros((h,w))
    for x in xrange(0, w):
        for y in xrange(0,h):
            arr[y][x] = pix[y,x]
    return _vectorize(arr)


def vec2im(im, vec):
    w,h = im.size
    arr = np.reshape(vec, (h,w))
    for x in xrange(0, w):
        for y in xrange(0, h):
            im.putpixel((y, x), arr[y][x])
    return im

