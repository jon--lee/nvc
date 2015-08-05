#c

from PIL import Image
import neuralpy2
import vectorizer
import numpy as np




class CompressorBase:

    # compress the image by propagating halfway through
    # the network (to the bottleneck) and returning the vector
    # at the halfway layer
    def compress(self, *args):
        raise NotImplementedError

    # expand the image by propagating from the middle layer
    # to the output layer which should reconstruct the image
    def expand(self, *args):
        raise NotImplementedError

    def forward(self, write_im):
        raise NotImplementedError

    def train(self, read_im):
        raise NotImplementedError

    def _vectorize(self, *args):
        raise NotImplementedError

    def _pix2vec(self, *args):
        






    def _vectorize(arr):
        h,w = arr.shape
        num = h*w
        return np.reshape(arr, (num))

    def _pix2vec(im):
        pix = im.load()
        w,h = im.size
        arr = np.zeros((h,w))
        for x in xrange(0, w):
            for y in xrange(0,h):
                arr[y][x] = pix[y,x]
        return _vectorize(arr)


    def _vec2im(im, vec):
        w,h = im.size
        arr = np.reshape(vec, (h,w))
        for x in xrange(0, w):
            for y in xrange(0, h):
                im.putpixel((y, x), arr[y][x])
        return im





def train(net, im):
    vec = vectorizer.pix2vec(im)
    max_ = net.start.size

    # split vector that represents the entire image into short divisions
    # that are easier to train. Together all the divisions in the list
    # represent the image
    training_inputs = [ vec[k*max_: (k+1)*max_] / 255 for k in xrange(0, len(vec) / max_) ]
    training_set = [(tvec, tvec) for tvec in training_inputs]

    net.train(training_set, 80, .3, mini_batch_size=5, monitor=True)
    # net.show_cost()

    return net    


def prop_image(net, im):
    output_vec = np.array([])
    for x, y in training_set:
        out = (net.forward(x).reshape((len(x))) * 255).astype(int)
        output_vec = np.concatenate((output_vec, out))

    im = vectorizer.vec2im(im, output_vec)
    return im

# assume this is black and white so pixels are not
# tuples of rgb values
im = Image.open('lena.bmp')

np.set_printoptions(threshold=np.nan)
net = neuralpy2.Network([64, 50, 64])
im = train(net, im)
im.show()
im.save('replicated_3.bmp')