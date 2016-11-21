import caffe
import numpy as np
import sys

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( '/home/th/Desktop/FYP_latest/BusDetector/FYP Alexnet/alexnet_jpg_50ep_001lr/mean.binaryproto' , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
print arr.shape
arr = np.reshape(arr, (3,256,256))
print arr.shape
np.save('/home/th/Desktop/FYP_latest/BusDetector/FYP Alexnet/alexnet_jpg_50ep_001lr/mean.npy', arr)