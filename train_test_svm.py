##
# Train SVM (or use the pretrained model) and then test.
# Pass the parameter retrain to train again
##
import itertools
import caffe
import numpy 
import pickle
import sys
import sklearn
from sklearn import svm
import os
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__=='__main__':
   USE_ONE_HOT = False

   train_file_path=str(sys.argv[1])
   test_file_path=str(sys.argv[2])
   labels={}

   if os.path.exists('trainedsvm.b'):
           with open('trainedsvm.b','rb') as model_file:
                model=pickle.load(model_file)
                print 'loaded pre-trained model'

   else:
           train_file=open(train_file_path,'rb')
           train_list=pickle.load(train_file)

           img_features=[]
           classes=[]
           for item in train_list:
                img_features.append(item[1])
                (name, _) = os.path.splitext(item[0])
                name_parts = name.split("_")
                item_class = []
                for name_part in name_parts:
                        try:
                                name_part = int(name_part)
                                item_class.append(name_part)
                        except ValueError:
                                continue
                classes.append(item_class[2])
                #print item_class[2]
           img_features=numpy.array(img_features)
           classes=numpy.array(classes)
           if USE_ONE_HOT:
                num_classes = numpy.amax(classes)
                one_hot_classes = numpy.zeros((len(classes), num_classes))
                for i in range(len(classes)):
                        one_hot_classes[i][classes[i]-1] = 1 
                print one_hot_classes.shape


           else:
                print classes.shape

           #print img_features.shape

           print 'Training svm model ...',
           sys.stdout.flush()
           model=sklearn.svm.SVC(kernel='linear',C=1,probability=True)
           if USE_ONE_HOT:
                   model.fit(img_features, one_hot_classes)
           else:
                   model.fit(img_features, classes)
           print 'done'


           with open('trainedsvm.b','wb') as model_file:
                   pickle.dump(model,model_file)


   test_file=open(test_file_path,'rb')
   test_list=pickle.load(test_file)

   img_features=[]
   classes=[]
   for item in test_list:
                img_features.append(item[1])
                (name, _) = os.path.splitext(item[0])
                name_parts = name.split("_")
                item_class = []
                for name_part in name_parts:
                        try:
                                name_part = int(name_part)
                                item_class.append(name_part)
                        except ValueError:
                                continue
                classes.append(item_class[2])
               # print item_class[2]
   img_features=numpy.array(img_features)
   classes=numpy.array(classes)
   if USE_ONE_HOT:
        num_classes = numpy.amax(classes)
        one_hot_classes = numpy.zeros((len(classes), num_classes))
        for i in range(len(classes)):
                one_hot_classes[i][classes[i]-1] = 1
        print one_hot_classes.shape
   else:
        print classes.shape

   print img_features.shape

   if USE_ONE_HOT:
        result = models.score(img_features, one_hot_classes)
   else:
        result = model.score(img_features, classes)
   print 'mean_accuracy in testing:' ,result
   
   


   conf_mat=numpy.zeros((6,6), dtype='int')

   for index in range(len(test_list)):
        res=model.predict([img_features[index,:]])
        if USE_ONE_HOT:
                print 'predicted:', res,'true_label:', one_hot_classes[index]
        else:
#               print 'predicted:', res,'true_label:', classes[index]
                conf_mat[classes[index],res]+=1



   plt.figure()
   plot_confusion_matrix(conf_mat, 
                classes=["soda_can","coffe_mug","cap","bowl","flashlight","cereal_box"],
                cmap=plt.cm.Blues)
   plt.show()
   savefig("depth_confusion_matrix.png")
   print conf_mat


