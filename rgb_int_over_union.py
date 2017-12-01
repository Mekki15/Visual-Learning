import pickle
import sys
from collections import namedtuple
import numpy as np
import cv2
import pickle
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score


truth_feat_file=str(sys.argv[1])
selective_feat_file=str(sys.argv[2])
models_file=str(sys.argv[3])
labels_file=str(sys.argv[4])


with open(truth_feat_file,'rb') as f:
    truth_features=pickle.load(f)

with open(selective_feat_file,'rb') as f:
    selective_features=pickle.load(f)

with open(models_file,'rb') as f:
    models_list=pickle.load(f)

with open(labels_file,'rb') as f:
    labels_list=pickle.load(f)

#print 'arooooo7',selective_features



def IOU(boxA,boxB):
    for i, bel in enumerate(boxB):
        boxB[i] = int(boxB[i])

        #(x,y) intersection
    #   boxA = [y1 x1 y2 x2]
    #   boxB = [x1 x2 y1 y2]
    if boxB[0]<boxA[1]<boxB[1] or boxB[0]<boxA[3]<boxB[1]:
      if boxB[2]<boxA[0]<boxB[3] or boxB[2]<boxA[2]<boxB[3]:
        xA = max(boxA[1], boxB[0])  #CHECK HOW OUR BBOX COORDINATES ARE SAVED        print 'XA' ,xA
        yA = max(boxA[0], boxB[2])
        xB = min(boxA[3], boxB[1])   #width
        yB = min(boxA[2], boxB[3])   #height
        #intersection area
        interArea=(xB - xA) * (yB - yA)
        #area of each bbox
        boxAArea =(boxA[3] - boxA[1]) *(boxA[2] - boxA[0])
        boxBArea =(boxB[1] - boxB[0]) *(boxB[3] - boxB[2])

        #iou computation
        iou = abs(interArea) / float(abs(boxAArea) +abs(boxBArea) - abs(interArea)) #we need to subtract the intersection area in order not to count it twice
        return iou

    return 0


# item[0] = imagename, item[1] = bboxes coordinates, item[2] = features
# feat[0] = imagename, feat[1] = classes, feat[2] = features, feat[3] = bboxes coordinates

if __name__== '__main__':
   detections=[]
   ## For each class (binary classifiers)
   for mod,label in zip(models_list,labels_list):
       Y_truth=[]
       Y_score=[]
       x_test=[]
       for item in selective_features:
	#don't take desk2,desk3,tablesmall2
#	 name=item[0].split('/')[-1]
#	 splt=name.split('_')
#	 if (splt[0]=='desk' and splt[1]=='1') or (splt[0]=='table' and splt[1]=='1') or (splt[0]=='table' and splt[1]=='small' and splt[2]=='1') or splt[0]=='kitchen' or splt[0]=='meeting':
#	  numer = name.replace('.png','').split('_')[-1]
#	  if int(numer)%5==0:
	    res=mod.predict(item[2])
            for feat in truth_features: 
	      ## if they refer to same image
	      if feat[0][0]==item[0]:
                 truth_boxes=feat[1]
                 selec_boxes=item [1]
                # if not 1 in res and label in feat[0][1]:
                  #  print label,'_false negative'
                 # for i,j in enumerate(feat[0][1]):
                  # if j==label: 
                   # Y_truth.append(1) #false_neagtives
                   # Y_score.append(0)
                   # x_test.append(feat[0][2][i])
#                    print item[0],feat[0][0]
                 #print feat[0][0]
                 #print item[0]  
                 if 1 in res and not label in feat[0][1]:
                   # print label,'_false positive'
                    for i,j in enumerate(res):
                      if int(j)==1: #false_positives
                        Y_truth.append(0)
                        Y_score.append(1)
                        x_test.append(item[2][i])
                      else:         #true_negatives
                        Y_truth.append(0)
                        Y_score.append(0)
                        x_test.append(item[2][i])
                 if  1 in res and  label in feat[0][1]:
                    prediction=[]
                    for i,j in enumerate(res):
                      if int(j)==1: 
                        predict=selec_boxes[i]
                        for x,y in enumerate(feat[0][1]):
                          if y==label:  
                             box=truth_boxes[x]
                            # IOU(predict,box)
                             if IOU(predict,box)>0.4: #true_positive
                               Y_truth.append(1)
                               Y_score.append(1)
                               x_test.append(item[2][i])        
                               prediction.append(item[0])
                               prediction.append(predict)
                             else:
                               Y_truth.append(0)
                               Y_score.append(1)
                               x_test.append(item[2][i])


      # print 'x_test is',np.array(x_test).shape
       y_score = mod.decision_function(x_test)
       average_precision = average_precision_score(Y_truth, y_score)

       print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))
       precision, recall, _ = precision_recall_curve(Y_truth, y_score)
       
       plt.clf()
       plt.step(recall, precision, color='b', alpha=0.2,
           where='post')
      
       plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

       plt.xlabel('Recall')
       plt.ylabel('Precision')
       plt.ylim([0.0, 1.05])
       plt.xlim([0.0, 1.0])
       plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
             average_precision))


#       fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
#       ax.plot() 
       plt.savefig('label'+str(label)+'.png')   # save the figure to file                      
                        #predictions.append(prediction)     
       



#                        selec_indices.append(i)
#                        selec_boxes.append(orig[1][i])
              
#              else:
#                   print 'nooo'

