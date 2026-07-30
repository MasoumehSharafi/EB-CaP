import cv2
import sys
import os
import csv
import pandas as pd

from src.detector import detect_faces
# from utils.visualization_utils import show_bboxes
from PIL import Image

#/home/gpa/Documents/Work/Data/RehabData/RehabData

def main():
	#col_list = ['subject_name', 'class_id', 'sample_name']
	#df = pd.read_csv('starting_point/samples.csv', sep='\t', usecols=col_list)
	#print(df)
	#sys.exit()
	root_dir = "Videos"
	total = 0 /
	count = 0
	for sub_dir in os.listdir(root_dir):
		sub_dir_path = os.path.join(root_dir, sub_dir)
		videos_list = os.listdir(sub_dir_path)
		# filter out "PA1" and "PA2" videos 
		videos_list = [video for video in videos_list if "PA1" not in video and "PA2" not in video]
		for file_dir in videos_list:
			print(os.path.splitext(file_dir)[0])
			file_dir_path = os.path.join(sub_dir_path, file_dir)
			cap = cv2.VideoCapture(file_dir_path)
			i = 2000
			while i < 5000:
				cap.set(cv2.CAP_PROP_POS_MSEC, i)
				success, image = cap.read()
				i = i + 40
				if success:
					pil_im = Image.fromarray(image)
					width, height = pil_im.size
					bounding_boxes, landmarks = detect_faces(pil_im)
					
					#print(bounding_boxes)
					if len(bounding_boxes) == 0:
						continue
					x, y, w, h, _ = bounding_boxes[0]
					if not os.path.isdir("sub_img_red_classes/"+ sub_dir):
						os.makedirs("sub_img_red_classes/"+ sub_dir)
					if not os.path.isdir("sub_img_red_classes/"+ sub_dir + "/" + os.path.splitext(file_dir)[0]):
						os.makedirs("sub_img_red_classes/"+ sub_dir + "/" + os.path.splitext(file_dir)[0])
					save_path = "sub_img_red_classes/"+ sub_dir + "/" + os.path.splitext(file_dir)[0]
					print(save_path)
				
					if (x>0 and y >0):
						cv2.imwrite(save_path + "/image" + str(i) + ".jpg", image[int(y) : int(h), int(x):int(w)]) 
					#success, image = cap.read()
					#cv2.imshow('frame', image) 

					#cv2.imwrite("images/"+ sub_dir + "/" + os.path.splitext(file_dir)[0] + "/" + "image" + str(i) + ".jpg", image) 
					#cv2.waitKey()
				


#		for file_path in os.listdir(file_dir_path):
#			file = os.path.join(file_dir_path, file_path)
#			save_path = "images" + "/" +sub_dir + "/" +file_dir
#			if not os.path.isdir(save_path):
#				os.makedirs(save_path)
#			count = count + 1 
#			total = total + 1

#		print(sub_dir)
#		print(count)
#		count = 0 
#	print(total)

#				image = cv2.imread(file)
#				pil_im = Image.fromarray(image)
#				print(file)
#				bounding_boxes, landmarks = detect_faces(pil_im)
#				print(bounding_boxes)
								
#				if len(bounding_boxes) == 0:
#					continue
#				x, y, w, h, _ = bounding_boxes[0]

				#print(bounding_boxes[0])
#				cv2.imwrite(save_path + "/" + os.path.splitext(file_path)[0] + ".jpg", image[int(y) : int(h), int(x):int(w)]) 
				#cv2.waitKey()
			

if __name__ == "__main__":
	main()
