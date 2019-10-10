# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:28:28 2019
@author: E442282
"""

import numpy as np
import cv2

from matplotlib import pyplot as plt

class Node(object):
	def __init__(self, value):
		self.value = value
		self.parent = self 
		self.rank = 0 
		
	def __str__(self):
		st = "[value: " + str(self.value) + ", parent: " + str(self.parent.value) 
		st += ", rank: " + str(self.rank) +  "]"
		return st
class union_find:
	# Constructor
	def __init__(self):
		self.__nodes_processed = {} 
	def createset(self, value):
		if self.getNode(value):
			return self.getNode(value)
		node = Node(value)
		self.__nodes_processed[value] = node
		return node
	def find(self, x):
		if x.parent  != x:  
			x.parent = self.find(x.parent) 
		return x.parent
	def union(self, x, y):
		if x == y:
			return
		x_root = self.find(x)
		y_root = self.find(y)
		if x_root == y_root:
			return
		if x_root.rank > y_root.rank: 
			y_root.parent = x_root
		elif x_root.rank < y_root.rank: 
			x_root.parent = y_root
		else: 
			x_root.parent = y_root
			y_root.rank = y_root.rank + 1
	def getNode(self, value):
		if value in self.__nodes_processed:
			return self.__nodes_processed[value]
		else:
			return False

def ccl(binary_image, connectivity_type=8):
	height,width = binary_image.shape[:2]
	labelled_image = np.zeros((height, width), dtype=np.int16)
	uf = union_find() 
	current_label = 1  
	
	for y, row in enumerate(binary_image):
		for x, pixel in enumerate(row):
			
			if pixel == False:
				# BG pixel -  0
				pass
			else: 
				# FG pixel - find the label
				labels = neighbouring_labels(labelled_image, connectivity_type, x, y)
				if not labels:
					uf.createset(current_label) # record label in disjoint set
					current_label = current_label + 1 # increment for next time				
				
				else:
					smallest_label = min(labels)
					labelled_image[y,x] = smallest_label
					if len(labels) > 1: 
										
						for label in labels:
							uf.union(uf.getNode(smallest_label), uf.getNode(label))
	final_labels = {}
	new_label_number = 1
	for y, row in enumerate(labelled_image):
		for x, pixel_value in enumerate(row):
			
			if pixel_value > 0: # FG 
			
				new_label = uf.Find(uf.getNode(pixel_value)).value 
				labelled_image[y,x] = new_label
				
				if new_label not in final_labels:
					final_labels[new_label] = new_label_number
					new_label_number = new_label_number + 1
	for y, row in enumerate(labelled_image):
		for x, pixel_value in enumerate(row):
			
			if pixel_value > 0: # Foreground pixel
				labelled_image[y,x] = final_labels[pixel_value]
	return labelled_image
def neighbouring_labels(image, connectivity_type, x, y):
	labels = set()
	if (connectivity_type == 4) or (connectivity_type == 8):
		# W
		if x > 0: 
			west_neighbour = image[y,x-1]
			if west_neighbour > 0: 
				labels.add(west_neighbour)
		# N
		if y > 0: # Pixel is not on top edge of image
			north_neighbour = image[y-1,x]
			if north_neighbour > 0: 
				labels.add(north_neighbour)
		if connectivity_type == 8:
			# NW
			if x > 0 and y > 0: 
				northwest_neighbour = image[y-1,x-1]
				if northwest_neighbour > 0: 
					labels.add(northwest_neighbour)
			# NE
			if y > 0 and x < len(image[y]) - 1: 
				northeast_neighbour = image[y-1,x+1]
				if northeast_neighbour > 0: 
					labels.add(northeast_neighbour)
	else:
		print("Wrong connectivity type")
	return labels        



def getColorSpaces(image):
    rgb = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    return rgb,gray

def getImageDimnesion(image):
    height,width = image.shape[:2]
    
    return height,width

def showImage(image,title,cmap):
    plt.imshow(image,cmap=cmap)
    plt.axis('off')
    plt.title(title)

def splitRGBChannels(image):
  red, green, blue= cv2.split(img)
  
  return red, green, blue

def getBinaryImage(gray):
    ret,thresh= cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    return thresh
 
def cropBorder(image,dx=20,dy=20):
    h,w=getImageDimnesion(image)    
    im=image.copy()
    im = im[dy:h-dy,dx:w-dx]
    
    return im
    
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
 
    
img_path=r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment4/input_data/text1.jpg'
img = cv2.imread(img_path)

img=cropBorder(img,20,20)

rgb,gray=getColorSpaces(img)

height,width=getImageDimnesion(img)

binary=getBinaryImage(gray)

plt.figure(figsize=(20, 20))

plt.subplot(3,2,1) 
plt.axis('off')
plt.title('Original-Crop border') 
plt.imshow(rgb)


output = ccl(binary, 8)

plt.subplot(3,2,2) 
plt.axis('off')
plt.title('CCL ')
plt.imshow(output,cmap='gray')









