# Visual-Learning
This README file explains the algorithm for the codes implemented in order to perform object detection on RGB-D data.
#
The dataset is the Washington RGB-D Scenes Dataset.It consist of eight video sequences of office and kitchen environments. 
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

1) First we train a separate classifier for each class of object using the hard negative mining procedure using the truth bboxes of the mat file. So at each step we retrain the misclassified examples (false positive) from the train dataset, remove the correctly classified examples and add some new examples, then we retrain the svm and so on. The procedure is described in the following:

    a. We take the features files generated in the Classification procedure and we train a SVM for each object for each data format(rgb, depth, merged): pos-neg-split_rgb.py, pos-neg-split_depth.py, pos-neg-split_merged.py/pos-neg-split_merged2.py .

2) Finally for testing we use selective search method to generate the agnostic bounding boxes. We use non maximum suppression to keep boxes with highest scores. We then plot the precision-recall curves and compare with the results from the paper that used the same dataset.
    
    b. We apply selective search algorithm to generate the predicted bounding-boxes on rgb images and we extract the features using Fast R-CNN network (extract_fc7_rgb_selective.py). 
    
    c. The bounding boxes are also used to extract objects from the depth images(crop_depth_ground_selective.py).
    
    d. These cropped images are used as an input into the DepthNet Network to extract depth features(extract_depth_features.py).

    e. The output features are then merged together (merge_features_selective.py).
    
    f. Classification is applied to each bounding-box detected : rgb_int_over_union.py, depth_int_over_union.py, merge_int_over_union.py. 
