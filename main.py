import neuralpy2
import numpy as np
import compressor
from PIL import Image

#np.set_printoptions(threshold=np.nan)


# assume this is black and white so pixels are not
# tuples of rgb values
im = Image.open('images/training/lena.bmp')
net = neuralpy2.Network([64, 50, 64])
com = compressor.Compressor(net)

com.train(im, 20, .3)

im2 = Image.open('images/testing/flower.bmp')
im2 = im2.convert('RGB').convert('P', palette=Image.ADAPTIVE)
im2 = com.forward(im2)

im2.show()
im2.save('images/products/lena_flower.bmp')