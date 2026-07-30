import cv2
import sys
import os
import csv

from src.detector import detect_faces
from utils.visualization_utils import show_bboxes
from PIL import Image

#file = "images/HI_PF07_S1/1/image38880.jpg"
#image = cv2.imread(file)
#crop_img = image[:, 0:350]
#pil_im = Image.fromarray(crop_img)
#width, height = pil_im.size
#bounding_boxes, landmarks = detect_faces(pil_im)
#print("done")
#sys.exit()

def main():
	root_dir = "images"
	total = 0
	count = 0

	subjects = ["EC_PF02_S1", "ES_PF03_S1", "JV_PF22_S1", "JV_PF22_S2", "GD_PF25_S1", "GD_PF25_S2", "KM_PF26_S1", "KM_PF26_S2", "KF_PF27_S1", "RL_PF28_S2", "RL_PF28_S1", "FD_PF29_S1", "FD_PF29_S2", "LD_PF30_S1", "LD_PF30_S2"]
	for subject in subjects:
		#sub_dir = "BEM_PF05_S2"
		sub_dir_path = os.path.join(root_dir, subject)
		for file_dir in os.listdir(sub_dir_path):
			file_dir_path = os.path.join(sub_dir_path, file_dir)
			print(file_dir)
			#if (file_dir == '2'):
			#	continue
			#if (file_dir == '1'):
			#	continue
			#if (file_dir == '3'):
			#	continue
			#if (file_dir == '5'):
                        #        continue
			for file_path in os.listdir(file_dir_path):
				print(file_path)
				file = os.path.join(file_dir_path, file_path)
				save_path = "faceimages" + "/" +subject + "/" +file_dir
				if not os.path.isdir(save_path):
					os.makedirs(save_path)
				#count = count + 1 
				#total = total + 1
				#print(sub_dir)
				#print(count)
				#count = 0 
				#print(total)
				crop_img = cv2.imread(file)
				#crop_img = image[:, 0:350]
				pil_im = Image.fromarray(crop_img)
				width, height = pil_im.size
				bounding_boxes, landmarks = detect_faces(pil_im)
				print(bounding_boxes)
				if len(bounding_boxes) == 0:
					continue
				x, y, w, h, _ = bounding_boxes[0]
				if (x>0 and y >0):
					#print(bounding_boxes[0])
					cv2.imwrite(save_path + "/" + os.path.splitext(file_path)[0] + ".jpg", crop_img[int(y) : int(h), int(x):int(w)]) 

if __name__ == "__main__":
	main()
