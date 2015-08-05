from PIL import Image
import neuralpy2
import vectorizer
import numpy as np

# assume this is black and white so pixels are not
# tuples of rgb values
im = Image.open('lena.bmp')
vec = vectorizer.pix2vec(im)

np.set_printoptions(threshold=np.nan)

net = neuralpy2.Network([64, 50, 64])

max_ = 64
training_inputs = [ vec[k*max_: (k+1)*max_] / 255 for k in xrange(0, len(vec) / max_) ]
len(training_inputs)

training_set = [(tvec, tvec) for tvec in training_inputs]
training_set = training_set[0:4096]
# print training_set
# print training_set[3][0][0:20]
net.train(training_set, 80, .3, mini_batch_size=5, monitor=True)
net.show_cost()



output_vec = np.array([])
for x, y in training_set:
    out = (net.forward(x).reshape((len(x))) * 255).astype(int)
    output_vec = np.concatenate((output_vec, out))

print output_vec[400:430]
print vec[400:430]

im = vectorizer.vec2im(im, output_vec)
im.show()
im.save('replicated_3.bmp')

# output_vec = np.array([])
# for vec in training_inputs:
#     output_vec = np.concatenate((output_vec, vec * 255))

# print len(output_vec)



# im = vectorizer.vec2im(im, output_vec)
# im.show()