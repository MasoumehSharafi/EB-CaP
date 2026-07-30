import cv2
import sys
import os
import csv
import pandas as pd

from PIL import Image

#/home/gpa/Documents/Work/Data/RehabData/RehabData

def main():
    # col_list = ['subject_name', 'class_id', 'sample_name']
    # df = pd.read_csv('starting_point/samples.csv', sep='\t', usecols=col_list)
    file = open("all_sub_labels.txt", "w")
    sub_class_file = open("sub_mapping_N_classes.txt", "w")
    # root_dir = 'sub_img_red_classes'
    # root_dir = 'sub_red_classes_img'
    root_dir = 'subject_images'
    class_count = 0
    
    target_list_name = ['081014_w_27','101609_m_36','112009_w_43','091809_w_43','071309_w_21','073114_m_25','080314_w_25','073109_w_28','100909_w_65','081609_w_40']
    
    for sub_dir in os.listdir(root_dir):
        # if sub_dir in target_list_name:
        #     continue
        sub_dir_path = os.path.join(root_dir, sub_dir)

        videos_list = os.listdir(sub_dir_path)

        # sub_class_file.write(sub_dir + "," + str(class_count) + "\n")
        # exclude_point = "101809_m_59"
        # videos_list = [video for video in videos_list if video != exclude_point]

		# filter out "PA1" and "PA2" videos 
		# videos_list = [video for video in videos_list if "PA1" not in video and "PA2" not in video]
        for file_dir in videos_list:
            sub_video_path = os.path.join(sub_dir_path, file_dir)
            # print(file_dir)
            vid_label = 0
            if "PA1" in file_dir :
                vid_label = 1
                # continue
            elif "PA2" in file_dir :
                vid_label = 2
                # continue
            elif "PA3" in file_dir :
                vid_label = 3
                # continue
            elif "PA4" in file_dir :
                vid_label = 4
            for sub_video in os.listdir(sub_video_path):
                write_file = os.path.join(sub_video_path, sub_video) + " " + str(vid_label)
                print(write_file)
                file.write(write_file + "\n")

                # --- Write: Treat subjects as a class 
                # write_file = os.path.join(sub_video_path, sub_video) + " " + (str(-1) if sub_dir in target_list_name else str(class_count))
                # print(write_file)
                # file.write(write_file + "\n")

        if sub_dir not in target_list_name:
            class_count = class_count + 1


            # print(label)
            # print(os.listdir(img_path))
    # sub_class_file.close()
    file.close()
	# for i in range(len(df['subject_name'])):
	# 	

if __name__ == "__main__":
	main()
