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
#import copy
from copy import deepcopy
import scipy.io
import skimage
import skimage.data
import selectivesearch
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy import *

def selective_boxes(filename):
          
          img=io.imread(filename)
          img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.8, min_size=10)
         # print len(regions)

          candidates=set()
          for r in regions:
              if r['rect'] in candidates:
                    continue
              if r['size'] < 1000:
                    continue

              x,y,w,h=r['rect']
              if w/h >1.2 or h/w >1.2:
                    continue
              candidates.add(r['rect'])

          fig, ax=plt.subplots(ncols=1, nrows=1, figsize=(6,6))
          ax.imshow(img)

          filterd_candidates = candidates.copy()
          for c in candidates:
               x, y, w, h = c

               for _x, _y, _w, _h in candidates:
                    if x == _x and y == _y and w == _w and h == _h:
                               continue

                    if abs(x - _x) < 10 and \
                       abs(y - _y) < 10 and \
                       w * h - _w * _h> 0:

                         filterd_candidates.discard((_x, _y, _w, _h))

#          print "candidates_length is", len(candidates)
#          print len(filterd_candidates)
          
          npboxes=[]
          for x,y,w,h in filterd_candidates:         
#             print x,y,w,h
             rect=mpatches.Rectangle((x,y),w,h, fill=False, edgecolor='red', linewidth=1)

             ax.add_patch(rect)
             boxes=[x,y,x+w,y+h]
             npboxes.append(boxes)
#         fig.savefig('img.png')
          return npboxes


def extract_features_boxes(folders):
  features = []
  num_im = 0
  num_image=0
  temp=1
  npboxes=[]
  for folder in folders: 
    for root,dirs,files in os.walk(folder):
       for file in files:
            filepath = root + "/" + file
            if 'png' in file and not 'depth' in file:
                im=cv2.imread(filepath)
                npboxes=selective_boxes(filepath)
                print "our boxes are",npboxes
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
                fc7 = deepcopy(net.blobs['fc7'].data)
                num_im += 1
               #print str(num_im),  "\b"*(len(str(num_im))+2),
                sys.stdout.flush()
                features.append((filepath,npboxes,deepcopy(fc7)))
	 	print features
  return features

def extract_features(folders):
        features = []
        num_im = 0
	for folder in folders:
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
