import numpy as np
import caffe
import os
from sklearn.svm import LinearSVC
import argparse as ap
import glob
import sys
import pickle
import random
from sklearn.calibration import CalibratedClassifierCV

if __name__=="__main__":
   neg_features={}
   pos_features={}
   features_file=str(sys.argv[1])
   model_list=str(sys.argv[2])
   label_list=str(sys.argv[3])
    #print Features_File

   with open(features_file,'rb') as f:
     features=pickle.load(f)
   print len(features)

   ## Create dict with labels as keys and corresponding features as values
   for item in features:
       image_name = item[0].replace(".png", "")
       label = int(image_name.split("_")[-1])
       print image_name, label
       feat = item[1]
       #label=item[1][0]
       if label in pos_features.keys():
            current=np.array(pos_features[label])
            next_feat=np.array([feat])
            pos_features[label]=np.concatenate((current,next_feat),axis=0)
       else:
            pos_features.update({label:[feat]})

   ## Create dict with labels as keys and non-corresponding features as values
   for key in pos_features.keys():
      for item in pos_features.keys():
        if not key==item:
           if key in neg_features.keys():
              current=neg_features[key]
              #next_feat=np.array([feat])
              next_feat=pos_features[item]
              neg_features[key]=np.concatenate((current,next_feat),axis=0)
           else:
              neg_features.update({key:pos_features[item]})

   labels=[]
   models_list=[]
   classes_names=[]

   for key in pos_features.keys():
      #labels=[]
      #img_feat=[]
      pos_feat=pos_features[key]
      pos_labels=np.ones(len(pos_feat),dtype=np.int)
      neg_feat=neg_features[key]
      neg_labels=np.zeros(len(neg_feat),dtype=np.int)
      img_feat=np.concatenate((pos_feat,neg_feat),axis=0)
      labels=np.concatenate((pos_labels,neg_labels),axis=0)

      classes_names.append(key)

      total=np.column_stack((img_feat,labels))
      print img_feat.shape, total.shape

      num_item = len(total)
      train_num =int(num_item*0.8)
      random.shuffle(total)

#      print total[:,:-1].shape
#      print total[:,-1]
      total_train_features = total[: train_num]
      train_features=total_train_features[:,:-1]

      train_labels=total_train_features[:,-1]
      total_test_features = total[train_num :]
      test_features=total_test_features[:,:-1]
      test_labels=total_test_features[:,-1]

      model=LinearSVC()
      #model=CalibratedClassifierCV(mod)
      model.fit(train_features,train_labels)


      result=model.score(test_features,test_labels) 
      print 'mean_accuracy in testing:', result

#      y_proba = model.predict_proba(test_features)
#      print y_proba
#      neg_hard_mining_feat=[]
#      neg_hard_mining_labels=[]

      for index in range(len(test_features)):
        res=model.predict([test_features[index,:]])
       # print "probability is" , res
#        print 'predicted:',res,'true_label:',test_labels[index]
#        if not res==test_labels[index]:
#          neg_hard_mining_feat.append(test_features[index,:])
#          neg_hard_mining_labels.append(test_labels[index])


#      print len(neg_hard_mining_feat),len(neg_hard_mining_labels)
#      model.fit(neg_hard_mining_feat,neg_hard_mining_labels)
#      result=model.score(test_features,test_labels)
#      print 'mean_accuracy in testing_after_hard_mining:' ,result



      models_list.append(model)


   with open(model_list,'wb') as f:
      model_list=pickle.dump(models_list,f)

   with open(label_list,'wb') as x:
      label_list=pickle.dump(classes_names,x)


#   with open(Test_Features,'rb') as f:
#     test_feat=pickle.load(f)

#   positives=[]
#   positive boxes={}
#   for mod in models_list:
#     for index in range(len(test_feat)):
#       res=models_list[0].predict(test_feat[index][2])
#       for i in res:
#           if int(res)==1.
#             positives.append 
#       print res
   #models_list[0]

