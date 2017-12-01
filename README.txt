##This README file explains the algorithm for the codes implemented in order to merge RGB and depth features

##The dataset is the  Washington RGB-D Scenes Dataset. It consist of eight video sequences of office and kitchen environments. 
  Each scene contain objects within the six categories bowl, cap, cereal box,coffee mug, soda can and flashlight.

##The Codes implemented for classification are as follows:

1) The first step is extracting RGB features from the dataset (extract_rgb_features.py). This is done by iterating through the matlab file
   associated with the dataset in order to extract the bounding boxes. 

2) The bounding boxes are then used as an input to the fast-rcnn network based on the pre trained model on pascal voc.

3) The bounding boxes are also used to extract objects from the depth images(crop_depth_ground_truth.py). 

4) These cropped images are used as an input into the DepthNet Network to extract depth features(extract_depth_features.py).

5) The output features from rgb and depth are then added to their labels in order to feed them to the svm (adjust_label_names_rgb.py , 
   adjust_label_names_depth.py)
  
6) The output features are then merged together (merge_features.py)

7) The features for rgb,depth and merged are then split in order to feed to the svm (split_features.py)

8) The features are then fed to the SVM(train_test_SVM.py)



##Detection Algorithm:

9) First we train a separate classifier for each class of object using the hard negative mining procedure using the truth bboxes of the mat file 
    So at each step we retrain the misclassified examples (false positive) from the train dataset, remove the correctly classified examples 
    and add some new examples, then we retrain the svm and so on.

10) Finally for testing we use selective search method to generate the agnostic bounding boxes.
    We use non maximum suppression to keep boxes with highest scores.
    We then plot the precision-recall curves and compare with the results from the paper that used the same dataset.
