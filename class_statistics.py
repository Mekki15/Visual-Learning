import caffe
import numpy 
import pickle
import sys
import sklearn
from sklearn import svm
import os


if __name__=='__main__':

   train_file_path=str(sys.argv[1])
   train_file=open(train_file_path,'rb')
   train_list=pickle.load(train_file)

   classes={}
   for item in train_list:
       (name, _) = os.path.splitext(item[0])
       name_parts = name.split("_")
       item_class = []
       for name_part in name_parts:
                try:
                        name_part = int(name_part)
                        item_class.append(name_part)
                except ValueError:
                        continue
       if item_class[2] in classes.keys():
                #print "Corresponding value", classes[item_class[2]]
                classes.update({item_class[2]: classes[item_class[2]]+1})
       else:
                classes.update({
                            item_class[2] : 1
                        })
   print classes
   sum = 0
   for i in range(len(classes.keys())):
        sum=sum+ classes[i]
   print sum
   for i in range(len(classes.keys())):
        print "percentage of class ", i,"is " , classes[i]*100/sum ,"%"


