import cv2
import numpy as np
import matplotlib.pyplot as plt
import caffe

print 'load files'
MODEL_FILE = "/home/th/Downloads/alexnet_models_new/deploy.prototxt"
PRETRAINED = "/home/th/Downloads/alexnet_models_new/snapshot_iter_300.caffemodel"
IMAGE_FILE = "/home/th/Desktop/bus_dataset_2/770/770_2_150.jpg"
print 'set mode'
caffe.set_mode_cpu()
print 'load pretrained'
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       channel_swap=(2, 1, 0),
                        mean=np.load("/home/th/Downloads/alexnet_models_new/a.npy"),
                       raw_scale=255,
                       image_dims=(256, 256))
input_image = caffe.io.load_image(IMAGE_FILE)
plt.imshow(input_image)

prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
print 'prediction shape:', prediction[0].shape
plt.plot(prediction[0])
print 'predicted class:', prediction[0].argmax()
plt.show()
