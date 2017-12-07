import pickle
import os
import sys
import numpy as np

depth_features = None
with open("../../features_depth_selective", "rb") as depth_file:
        depth_features = pickle.load(depth_file)

rgb_features = None
with open("../../rgb_selective_features", "rb") as rgb_file:
        rgb_features = pickle.load(rgb_file)

if depth_features and rgb_features:
        merged_features = {}
        prev_f = None
        depth_features_boxes = {}
	for depth_feature in depth_features:
#		print 'Depth feature: ', depth_feature
		depth_original_path = os.path.split(depth_feature[0])[0]
                depth_original_name = os.path.split(depth_feature[0])[1]
                depth_original_name = depth_original_name.replace("_depth", "")
                bbox_count=depth_original_name.split("_")[-1]
                depth_name= depth_original_name.replace("_"+bbox_count,".png") #Image name
                bbox_count= bbox_count.replace(".png","") #Bbox index
               	d_feat=depth_feature[1]
		index=int(bbox_count)-1

		for rgb_feature in rgb_features:
			crop_name=os.path.split(rgb_feature[0])[1]
			if crop_name == depth_name:
        	                if "meeting_small_1_27" in depth_name:
	                                print depth_name, index
				r_feat=rgb_feature[2][index]
#				print 'r_feat ',r_feat
				bbox = rgb_feature[1][index]
				m_feat=np.concatenate((d_feat,r_feat),axis=0) #merged feature vector
				merged_features.update({depth_original_name : (m_feat, bbox)})
				full_path = os.path.join(depth_original_path, depth_name)
				if full_path in depth_features_boxes.keys():
					depth_features_boxes[full_path].insert(index, (bbox, depth_feature[1]))
					if "meeting_small_1_27" in depth_name:
						print "inserted", index
				else:
					depth_features_boxes.update({
						full_path : []
					})
					depth_features_boxes[full_path].insert(index, (bbox, depth_feature[1]))
					if "meeting_small_1_27" in depth_name:
						print "created vector", len(rgb_feature[1]), index
		# if depth_crop_name in merged_features.keys():
                 #       data=merged_features[depth_crop_name]
                  #      bbox_index= int(bbox_count)-1
                   #     print 'Name:',depth_crop_name, 'data', data, "\t", bbox_index, "\t", data[bbox_index]
                    #    merged_features[depth_crop_name]= np.concatenate((data[bbox_index],depth_feature[1]),axis=0)




#        for rgb_feature in rgb_features:
#		feature=[]
 #               crop_name = os.path.split(rgb_feature[0])[1]
#		feature.append((rgb_feature[1],rgb_feature[2]))
  #              merged_features.update({crop_name:rgb_feature[2]})
#       	for depth_feature in depth_features:
#               	depth_crop_name = os.path.split(depth_feature[0])[1]
#                depth_crop_name = depth_crop_name.replace("_depth", "")
#		bbox_count=depth_crop_name.split("_")[-1]
#		depth_crop_name= depth_crop_name.replace("_"+bbox_count,".png")
#		bbox_count= bbox_count.replace(".png","")
#		if depth_crop_name in merged_features.keys():
#			data=merged_features[depth_crop_name]
#			bbox_index= int(bbox_count)-1
#			print 'Name:',depth_crop_name, 'data', data, "\t", bbox_index, "\t", data[bbox_index]
 #               	merged_features[depth_crop_name]= np.concatenate((data[bbox_index],depth_feature[1]),axis=0)
#        prev_f = None
#        print "Merged ", len(merged_features.keys()), "crops"
#        print "Merged feature vector size: ", merged_features[merged_features.keys()[0]].shape

        #print "Saving merged features to merged_fc7_features_new ...",
        #sys.stdout.flush()
        #with open("merged_fc7_features_selective", "wb") as merged_file:
        #        pickle.dump(merged_features.items(), merged_file)
        #print "DONE"

	print "Saving depth with also bboxes...",
	sys.stdout.flush()
	with open("features_depth_selective_boxes", "wb") as depthb_file:
		pickle.dump(depth_features_boxes, depthb_file)
	print "DONE"
