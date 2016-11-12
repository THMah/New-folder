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
MODEL_FILE = "/home/th/Downloads/alexnet_models_3bus_100echo/deploy.prototxt"
PRETRAINED = "/home/th/Downloads/alexnet_models_3bus_100echo/snapshot_iter_1800.caffemodel"
IMAGE_FOLDER = "/home/th/Desktop/bus_771_test_caffe"
LIVE_VIDEO_FRAME = '/home/th/Desktop/FYP_latest/New-folder/Live_video_frame'

print 'capture video'
#use 0 to capture live video / VideoPath to play local video
cap = cv2.VideoCapture("/home/th/Desktop/bus video oct/Editted/bus_771.mp4")
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
                                #RGB to BGR
                               channel_swap=(2, 1, 0),
                               raw_scale=255,
                               image_dims=(227, 227))
        input_image = caffe.io.load_image(LIVE_VIDEO_FRAME + '/' + file)
        plt.imshow(input_image)
        #predict takes any number of images, and formats them for the Caffe net automatically
        prediction = net.predict([input_image])
        #print 'prediction shape:', prediction[0].shape
        #plt.plot(prediction[0])
        output = ["770", "771", "SJ01", "No bus"]
        print 'predicted class:', output[prediction[0].argmax()]
        str1 = 'bus' + str(output[prediction[0].argmax()])
        #plt.text(2, 1, str1, fontsize=15)
        plt.suptitle(str1, fontsize=15)
        plt.show()