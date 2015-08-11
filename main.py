import neuralpy2
import numpy as np
import compressor
from PIL import Image

#np.set_printoptions(threshold=np.nan)


def new_image(path):
    im = Image.open(path)
    im = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)

    return im

# assume this is black and white so pixels are not
# tuples of rgb values
im = new_image('images/training/lena.bmp')
im2 = new_image('images/testing/flower.bmp')

#net = neuralpy2.Network([64, 50, 64])
#com = compressor.Compressor(net)
com = compressor.Compressor([64, 50, 64])

com.train([im, im2], 20, .3)

im_new = new_image('images/training/peppers.bmp')
im_new = com.forward(im_new)

im_new.show()
im_new.save('images/products/peppers-from-multiple.bmp')
