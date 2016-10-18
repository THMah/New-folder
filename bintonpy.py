import caffe
import numpy as np
import sys

c=np.fromfile("/home/th/Downloads/alexnet_models/mean.binaryproto", dtype=np.float)
print c
np.save("a.npy", c)
b=np.load('a.npy')
print b

print "success"