
import scipy.io
import os
import sys
import pickle


with open ('datasets/features/rgb_fc7_features','rb') as feat:
    features=pickle.load(feat)

labels_list= open ('datasets/rgbd-scenes/labels.txt','rb')
for item in labels_list:
    print item

mat_list=[]

for root,_,files in os.walk('datasets/rgbd-scenes'):
      for file in files:
        if 'mat' in file:
           print file
           mat_list.append(os.path.join(root,file))
class_x=""

new_features=[]
for feature in features:
   name=feature[0].split('/')[3]
   num=os.path.split(feature[0])[1].replace('.png','').split('_')[-1]
   
   for item in mat_list:
     if name in item:
        mat_file=scipy.io.loadmat(item)
   print"old feature is", feature
   boxes=mat_file['bboxes']
   print "our boxes are" ,boxes[0][int(num)-1][0][0][0][0]
   x=boxes[0][int(num)-1].shape[1]
   if not x==0:
     for i in range(x):
       class_x = boxes[0][int(num)-1][0][i][0][0]
       print "elclass_x",class_x

       labels_list= open ('datasets/rgbd-scenes/labels.txt','rb')
       for label in labels_list:
          if class_x in label:
            print label[1] 
            feature[1][i]=int(label[0])

   new_features.append(feature)         
   print "new feature is " ,feature  
#   print "our features are" , feature[1]
with open('feature_file_new', 'wb') as f:
         pickle.dump(new_features,f)   


