import neuralpy2
import numpy as np
import compressor
from PIL import Image

#np.set_printoptions(threshold=np.nan)


# assume this is black and white so pixels are not
# tuples of rgb values
im = Image.open('images/training/lena.bmp')
im2 = Image.open('images/training/peppers.bmp')
net = neuralpy2.Network([64, 50, 64])
com = compressor.Compressor(net)

com.train([im, im2], 20, .3)

im_new = Image.open('images/testing/flower.bmp')
im_new = im_new.convert('RGB').convert('P', palette=Image.ADAPTIVE)
im_new = com.forward(im_new)

im_new.show()
im_new.save('images/products/flower-from-multiple.bmp')
