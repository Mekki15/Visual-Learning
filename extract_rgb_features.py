##
# You must provide in input the folder with the images, the folder has to contains a list
# of subfolders, one for each class of images.
# The features will be saved in the file "image-features/CLASS_NAME/features" for each class.
##
import sys
sys.path.append("py-faster-rcnn/tools")
import _init_paths
from fast_rcnn.test import _get_blobs
from fast_rcnn.config import cfg

import numpy as np
import caffe
import os
import glob
import scipy.misc
import pickle
import cv2
from numpy import newaxis
import copy
import scipy.io

def extract_features_boxes(folders):
  features = []
  num_im = 0
  num_image=0
  temp=1
  npboxes=[]
  for folder in folders: 
    for root,dirs,files in os.walk(folder):
       for file in files:
         if 'mat' in file:
           matf=scipy.io.loadmat(root+'/'+file)
           boxes=matf['bboxes']
           new_root=root+'/'+file.split('.')[0]
           for _,_,files in os.walk(new_root):
             for file in files:
                filepath = new_root + "/" + file
                if 'png' in file and not 'depth' in file:
                   #im = caffe.io.load_image(filepath)
                   im = cv2.imread(filepath)
                   nx=os.path.splitext(root+'/'+file)[0]
                   num= int(nx.split('_')[-1])
                   #num_image=int(nx.split('_')[-2])
                   print num
                   #if not num_image == temp
                   npboxes=[]
                   labels_list=[]
                   #if not len(boxes[0][num-1][0])==0:
                   x=boxes[0][num-1].shape[1]
                   #print x ," is x"
                   if not  x == 0:
                     for i in range(x):
                        bboxes_list=[]
                        bboxes_list.append(boxes[0][num-1][0][i][4][0][0]) #x1
                        bboxes_list.append(boxes[0][num-1][0][i][2][0][0]) #y1
                        bboxes_list.append(boxes[0][num-1][0][i][5][0][0]) #x2
                        bboxes_list.append(boxes[0][num-1][0][i][3][0][0]) #y2

                        npboxes.append(bboxes_list)
                        labels_list.append(boxes[0][num-1][0][i][1][0][0])
                     in_blobs, _ = _get_blobs(im, np.asarray(npboxes))
                     net.blobs['data'].reshape(*(in_blobs["data"].shape))
                     net.blobs['rois'].reshape(*(in_blobs["rois"].shape))
                     blobs_out=net.forward(data=in_blobs["data"].astype(np.float32,copy=False),
                                          rois=in_blobs["rois"].astype(np.float32, copy=False))
                    # print "OUR: ", transformer.preprocess("data", im).shape
                    # print "DEMO: ", in_blobs["data"].shape
                     print file
                     print blobs_out.keys()
                     print blobs_out["cls_prob"]
                    # net.blobs['rois'].reshape(len(npboxes),5)
                    # net.reshape()
                    # net.blobs['data'].data[...] = transformer.preprocess('data', im)
                    # net.blobs['rois'].data[...] = np.array(npboxes)
                    # out=net.forward()
                     fc7 = copy.deepcopy(net.blobs['fc7'].data)
                     num_im += 1
                     #print str(num_im),  "\b"*(len(str(num_im))+2),
                     sys.stdout.flush()
                     features.append((filepath, labels_list, copy.deepcopy(fc7)))
                     print fc7, npboxes, labels_list
  return features

def extract_features(folders):
        features = []
        num_im = 0
        for folder in folders:
                print folder+"eh elkalam"
                for root,dirs, files in os.walk(folder):
                        #mat=scipy.io.loadMat(root+*.Mat)
                        for file in files:
                            filepath = root + "/" + file
                            if 'png' in file:
                                im = caffe.io.load_image(filepath)
                                #im = im[:,:,0]

                                # scipy.misc.imsave(filename + 'mod.png', im)
                                #im = im.reshape((im.shape[0], im.shape[1], 1))

                                net.blobs['data'].data[...] = transformer.preprocess('data', im)
                                out = net.forward()
                                fc7 = net.blobs['fc7'].data[0]
                                num_im += 1
                                print str(num_im),  "\b"*(len(str(num_im))+2),
                                sys.stdout.flush()
                                features.append((filepath, copy.deepcopy(fc7)))
        return features

if __name__ == '__main__':
        folder = str(sys.argv[1])
        feature_file = str(sys.argv[2])

        net = caffe.Net("py-faster-rcnn/models/pascal_voc/VGG16/fast_rcnn/test.prototxt",
                        "py-faster-rcnn/data/fast_rcnn_models/vgg16_fast_rcnn_iter_40000.caffemodel",
                        caffe.TEST)

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        meanx = np.load(os.path.join(folder, "mean_image.npy"))
        print meanx.shape
        mean_vec=cfg.PIXEL_MEANS.reshape((3))
        print "mean_vec shape:", mean_vec.shape
        transformer.set_mean('data', mean_vec)
        #transformer.set_raw_scale('data', 255.0)

        net.blobs['data'].reshape(1,3,224,224)

        features=extract_features_boxes([folder])
        with open(feature_file, 'wb') as f:
                pickle.dump(features,f)

