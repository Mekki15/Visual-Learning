
import pickle
import os
import sys
import numpy as np

depth_features = None
with open("../features/depth_fc7_features_new", "rb") as depth_file:
        depth_features = pickle.load(depth_file)

rgb_features = None
with open("../features/rgb_fc7_features_new_v2", "rb") as rgb_file:
        rgb_features = pickle.load(rgb_file)

if depth_features and rgb_features:
        merged_features = {}
        prev_f = None
        for depth_feature in depth_features:
                crop_name = os.path.split(depth_feature[0])[1]
                crop_name = crop_name.replace("_depth", "")
                merged_features.update({
                            crop_name : depth_feature[1]
                        })
                if prev_f: assert(prev_f==depth_feature[1])
        prev_f = None
        for rgb_feature in rgb_features:
                crop_name = os.path.split(rgb_feature[0])[1]
                if crop_name in merged_features.keys():
                        merged_features[crop_name] = np.concatenate((
                                    merged_features[crop_name],
                                    rgb_feature[1]
                                ))
                if prev_f: assert(prev_f==rgb_feature[1])
        print "Merged ", len(merged_features.keys()), "crops"
        print "Merged feature vector size: ", merged_features[merged_features.keys()[0]].shape

        print "Saving merged features to merged_fc7_features_new ...",
        sys.stdout.flush()
        with open("merged_fc7_features_new", "wb") as merged_file:
                pickle.dump(merged_features.items(), merged_file)
        print "DONE"
#       print os.path.split(rgb_features[0][0]), os.path.split(depth_features[0][0])


