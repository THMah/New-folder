import cv2
import numpy as np
import matplotlib.pyplot as plt
import caffe
import glob
import os


print 'load files'
MODEL_FILE = "/home/th/Downloads/alexnet_models_3bus_100echo/deploy.prototxt"
PRETRAINED = "/home/th/Downloads/alexnet_models_3bus_100echo/snapshot_iter_1800.caffemodel"
IMAGE_FILE = "/home/th/Desktop/bus_dataset/SJ01/bus_771_275.jpg"
VIDEO_FILE = "/home/th/Desktop/bus video oct/Editted/bus_771.mp4"
VIDEO_FILE1 = glob.glob("/home/th/Desktop/bus video oct/Editted/bus_771.mp4")
IMAGE_FILE1 = "/home/th/Desktop/bus_771_test_caffe"

for file in os.listdir(IMAGE_FILE1):
    if file.endswith(".jpg"):
        print (IMAGE_FILE1 + '/' + file)

        print 'set mode'
        #use GPU to do calculation
        caffe.set_mode_gpu()

        #use CPU to do calculation
        #caffe.set_mode_cpu()

        print 'load pretrained'
        net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                                #RGB to BGR
                               channel_swap=(2, 1, 0),
                               raw_scale=255,
                               image_dims=(227, 227))
        print 'input image'
        input_image = caffe.io.load_image(IMAGE_FILE1 + '/' + file)
        print 'show image'
        plt.imshow(input_image)

        #predict takes any number of images, and formats them for the Caffe net automatically
        prediction = net.predict([input_image])
        #print 'prediction shape:', prediction[0].shape
        plt.plot(prediction[0])
        output = ["770", "771", "SJ01", "No bus"]
        print 'predicted class:', output[prediction[0].argmax()]
        plt.show()