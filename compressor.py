from PIL import Image
import neuralpy2
import numpy as np




class CompressorBase:

    # initializer should take in a network that is used
    # to compress and expand the data
    def __init__(self, *args):
        raise NotImplementedError

    # compress the image by propagating halfway through
    # the network (to the bottleneck) and returning the vector
    # at the halfway layer
    def compress(self, *args):
        raise NotImplementedError

    # expand the image by propagating from the middle layer
    # to the output layer which should reconstruct the image
    def expand(self, *args):
        raise NotImplementedError

    # propagate the image all the way forward.
    # this function is primarily used for debuggin
    # a trained replicator
    def forward(self, write_im):
        raise NotImplementedError

    # train the network to recognize a set of images
    # images must be vectorized and concatenated in order
    # to properly train the network It is assumed that the
    # network input size divides evenly into each image
    def train(self, read_ims):
        raise NotImplementedError

    # vectorize an matrix by collapsing
    # the matrix into a single vector
    def _vectorize(self, *args):
        raise NotImplementedError

    # convert an PIL image object to an expanded vector
    def _im2vec(self, *args):
        raise NotImplementedError

    # convert an expanded vector to a PIL image object
    def _vec2im(self, *args):
        raise NotImplementedError






class Compressor:

    def __init__(self, net):
        self.net = net


    def train(im, epochs, learning_rate):
        vec = self._im2vec(im)
        max_ = self.net.start.size

        # split vector that represents the entire image into short divisions
        # that are easier to train. Together all the divisions in the list
        # represent the image
        training_inputs = [ vec[k*max_: (k+1)*max_] / 255 
            for k in xrange(0, len(vec) / max_) ]
        training_set = [(tvec, tvec) for tvec in training_inputs]

        net.train(training_set, epochs, learning_rate, mini_batch_size=5, monitor=True)
        return net    


    def forward(self, im):
        output_vec = np.array([])
        for x, y in training_set:
            out = (net.forward(x).reshape((len(x))) * 255).astype(int)
            output_vec = np.concatenate((output_vec, out))

        im = self._vec2im(im, output_vec)
        return im


    def _vectorize(self, arr):
        h,w = arr.shape
        num = h*w
        return np.reshape(arr, (num))


    def _im2vec(self, im):
        pix = im.load()
        w,h = im.size
        arr = np.zeros((h,w))
        for x in xrange(0, w):
            for y in xrange(0,h):
                arr[y][x] = pix[y,x]
        return self._vectorize(arr)


    def _vec2im(self, im, vec):
        w,h = im.size
        arr = np.reshape(vec, (h,w))
        for x in xrange(0, w):
            for y in xrange(0, h):
                im.putpixel((y, x), arr[y][x])
        return im

