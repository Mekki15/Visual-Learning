
import numpy as np
import pickle
import random
import sys
import os

if __name__ == '__main__':

        features_path_file = str(sys.argv[1])

        orig_features= None
        with open(features_path_file, "rb") as orig_file:
                orig_features = pickle.load(orig_file)
        print len(orig_features)        
        filtered_features=[]
        for item in orig_features:
           num_list=[]
           elems=os.path.split(os.path.splitext(item[0])[0])[-1].split('_')
           for elem in elems:
             try: 
               int(elem)
               num_list.append(int(elem))
             except:
               continue
           if not len(num_list) ==0:
             if num_list[1]%5==0:
               filtered_features.append(item)


        random.shuffle(filtered_features)
        num_item = len(filtered_features)

        print "Total num items:", num_item

        train_num =int(num_item*0.8)

        train_features = filtered_features[: train_num]
        test_features = filtered_features[train_num :]

        print "Num train items:", len(train_features)
        print "Num test items:", len(test_features)
        with open(features_path_file + "_train", "wb") as train_file:
                pickle.dump(train_features, train_file)

        with open(features_path_file + "_test", "wb") as test_file:
                pickle.dump(test_features, test_file)


