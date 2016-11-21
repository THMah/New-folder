import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import os
#disable caffe log output :
#0 - debug
#1 - info (still a LOT of outputs)
#2 - warnings
#3 - errors
os.environ['GLOG_minloglevel'] = '2'
import caffe



print 'load files'
MODEL_FILE = "/home/th/Desktop/FYP_latest/BusDetector/FYP Alexnet/alexnet_jpg_30ep_001lr/deploy.prototxt"
PRETRAINED = "/home/th/Desktop/FYP_latest/BusDetector/FYP Alexnet/alexnet_jpg_30ep_001lr/snapshot_iter_330.caffemodel"
LIVE_VIDEO_FRAME = '/home/th/Desktop/FYP_latest/BusDetector/Live_video_frame'

print 'capture video'
#use 0 to capture live video / VideoPath to play local video
cap = cv2.VideoCapture("/home/th/Desktop/FYP_latest/BusDetector/bus video oct/Editted/bus_sj01_4.mp4")
#frame rate
frameRate = cap.get(5)
while(cap.isOpened()):
#current frame number
    frameId = cap.get(1)
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        resizeImage = cv2.resize(frame, (256, 256))
        now = format(datetime.datetime.now())
        filename = LIVE_VIDEO_FRAME + "/" + str(now) + ".jpg"
        cv2.imwrite(filename, resizeImage)
cap.release()
#end capture video and save as image

for file in sorted(os.listdir(LIVE_VIDEO_FRAME)):
    if file.endswith(".jpg"):
        print(file)
        print 'set mode'
        #use GPU to do calculation
        caffe.set_mode_gpu()

        #use CPU to do calculation
        #caffe.set_mode_cpu()

        print 'load pretrained'
        net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                                image_dims=(227, 227),
                                raw_scale=255,
                                 mean=np.load("/home/th/Desktop/FYP_latest/BusDetector/FYP Alexnet/alexnet_jpg_30ep_001lr/mean.npy").mean(1).mean(1),
                                #RGB to BGR
                                #caffe input expected BGR input
                               channel_swap=(2, 1, 0))
        #this function load img in RGB
        input_image = caffe.io.load_image(LIVE_VIDEO_FRAME + '/' + file)
        plt.imshow(input_image)
        #predict takes any number of images, and formats them for the Caffe net automatically
        prediction = net.predict([input_image])
        print(prediction)
        #print 'prediction shape:', prediction[0].shape
        #plt.plot(prediction[0])
        output = ["771", "SJ01", "No bus"]
        print 'predicted class:', output[prediction[0].argmax()]
        str1 = 'bus: ' + str(output[prediction[0].argmax()])
        #plt.text(2, 1, str1, fontsize=15)
        plt.suptitle(str1, fontsize=15)
        plt.show()
