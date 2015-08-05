from PIL import Image
import numpy as np
im = Image.open('lena.bmp')
im = im.resize((100, 100))
pix = im.load()
w,h = im.size

quarter = 256 / 8

arr = [[""]*h]*w
s = ""
for x in xrange(0, h):
    for y in xrange(0, w):
        s+= " "
        gray = pix[y,x]
        #gray = (r + g + b) /3
        if gray < quarter:
            s+=""
        elif gray < quarter * 2:
            s+="-"
        elif gray < quarter * 3:
            s+="+"
        elif gray < quarter * 4:
            s+="="
        elif gray < quarter * 5:
            s+="o"
        elif gray < quarter * 7:
            s+="G"
        else:
            s+="#"
    print s
    s = ""
