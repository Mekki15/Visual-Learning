
##
# You must provide in input the folder with the images, the folder has to contains a list
# of subfolders, one for each class of images.
# The features will be saved in the file "image-features/CLASS_NAME/features" for each class.
##

import numpy as np
import caffe
import os
import sys
import glob
import scipy.misc
import pickle
import cv2
from numpy import newaxis
import copy

def extract_features(folders):
        features = []
        num_im = 0
        for folder in folders:
                print folder
                for root, _, files in os.walk(folder):
                        for file in files:
                            filepath = root + "/" + file
                            if 'png' in file:
                                im = caffe.io.load_image(filepath)
                                im = im[:,:,0]

                                # scipy.misc.imsave(filename + 'mod.png', im)
                                im = im.reshape((im.shape[0], im.shape[1], 1))

                                net.blobs['data'].data[...] = transformer.preprocess('data', im)
                                out = net.forward()
                                fc7 = net.blobs['fc7'].data[0][:]
                                num_im += 1
#                               print str(num_im),  "\b"*(len(str(num_im))+2),
                                sys.stdout.flush()

                                features.append((filepath, copy.deepcopy(fc7)))
                                print file
                                print np.argmax(out["softmax"])
        return features

if __name__ == '__main__':
        folder = str(sys.argv[1])
        feature_file = str(sys.argv[2])
#       img_feat_dir = str(sys.argv[2])
        #img_feat_dir = "image-features"

        net = caffe.Net("depthNet/deploy.txt",
                        "depthNet/vandal_washington_snapshot.caffemodel",
                        caffe.TEST)

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        meanx = np.load(os.path.join(folder, "mean_image.npy"))[0]
        meanx = np.expand_dims(meanx, axis=0)
        print meanx.shape
        mean_vec=meanx.mean(1).mean(1)
        print "mean_vec shape:", mean_vec.shape
        transformer.set_mean('data', mean_vec)
        transformer.set_raw_scale('data', 255.0)

        net.blobs['data'].reshape(1,1,227,227)

        features=extract_features([folder])
        with open(feature_file, 'wb') as f:
                pickle.dump(features,f)

