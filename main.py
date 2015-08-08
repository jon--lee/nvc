import neuralpy2
import numpy as np
import compressor
from PIL import Image

#np.set_printoptions(threshold=np.nan)


# assume this is black and white so pixels are not
# tuples of rgb values
im = Image.open('peppers.bmp')
net = neuralpy2.Network([64, 50, 64])
com = compressor.Compressor(net)

com.train(im, 10, .3)
im = com.forward(im)

im.show()
im.save('replicated_3.bmp')