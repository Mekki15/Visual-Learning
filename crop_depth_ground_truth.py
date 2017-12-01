
import cv2
import scipy.io
import os

def choose_file_name(orig_name, class_num):
        if os.path.isfile(orig_name):
            (name, ext) = os.path.splitext(orig_name)
            components = name.split("_")
            name = "_".join(components[:len(components)-1])
            orig_name = name + "_" + class_num + "_depth" + ext
            orig_name = choose_file_name(orig_name, class_num)
        return orig_name

def crop_ims(mat_file_path, ims_folder_path, crop_output_folder):
        print "root: ", ims_folder_path
        bbs = scipy.io.loadmat(mat_file_path)['bboxes'][0]
        num_ims = len(bbs)
        (root, ims_folder) = os.path.split(ims_folder_path)
        ## for each image
        for i, bb in enumerate(bbs):
                im_base_name = ims_folder + "_" + str(i+1)
                im_path = os.path.join(ims_folder_path, im_base_name + "_depth.png")
                img = cv2.imread(im_path)
                ## for each ROI in image
                for roi in bb[0]:
                        class_num = str(roi[1][0][0])
                        x1 = roi[2][0][0]
                        x2 = roi[3][0][0]
                        y1 = roi[4][0][0]
                        y2 = roi[5][0][0]
                        crop_im = img[x1:x2, y1:y2]
                        output_file_name = os.path.abspath(os.path.join(crop_output_folder, im_base_name + "_" + class_num + "_depth.png"))
                        output_file_name = choose_file_name(output_file_name, class_num)
                        cv2.imwrite(output_file_name, crop_im)



if __name__ == "__main__":

        output_folder = "../cropped-rgbd-depth-scenes/"
        root_folder = "../rgbd-scenes/"
        for root, _, files in os.walk(root_folder):
                for file in files:
                        (name, ext) = os.path.splitext(file)
                        if ext == ".mat":
                                mat_file_path = os.path.join(root, file)
                                ims_folder = os.path.join(root, name)
                                crop_output_folder = os.path.join(output_folder, name)
                                os.system("mkdir -p " + crop_output_folder)
                                crop_ims(mat_file_path, ims_folder, crop_output_folder)


