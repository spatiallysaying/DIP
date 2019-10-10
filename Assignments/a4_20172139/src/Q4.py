import numpy as np
import cv2

from matplotlib import pyplot as plt

import scipy.ndimage.morphology as m

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
 


def getSquareObjects(binary):    
    image, contours, hierarchy  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
    mask = np.zeros_like(img)
    img_copy=img.copy()
            
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
#            cv2.drawContours(img_copy, [cnt], -1, (255,0,255), 2)
            cv2.fillPoly(mask, pts =[cnt], color=(255,255,255))
            mask = cv2.bitwise_and(img_copy, mask)
    
    rgb,gray=getColorSpaces(mask)
    square_objects=getBinaryImage(gray)
    
    return  square_objects       

def getCircleObjects(binary):    
    image, contours, hierarchy  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
    mask = np.zeros_like(img)
    img_copy=img.copy()
            
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        if len(approx) > 4:
#            cv2.drawContours(img_copy, [cnt], -1, (255,0,255), 2)
            cv2.fillPoly(mask, pts =[cnt], color=(255,255,255))
            mask = cv2.bitwise_and(img_copy, mask)
    
    rgb,gray=getColorSpaces(mask)
    circle_objects=getBinaryImage(gray)
    
    return  circle_objects   


def getSqaureCircleCount(binary):
    rect_count=0
    circle_count=0
    image, contours, hierarchy  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
           rect_count=rect_count+1
        else:    
           circle_count=circle_count+1
    return    rect_count,    circle_count
   
def getShape(contour):    
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    if len(approx) == 4:
       return 1
    else:    
       return 2
    
    
def getObjectsWithHoles(binary):    
    image, contours, hierarchy  = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
    mask = np.zeros_like(img)
    img_copy=img.copy()    
    count_with_holes=0
    for component in zip(contours, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]
        x,y,w,h = cv2.boundingRect(currentContour)
#        if currentHierarchy[2] < 0:
#            #Single hierarchy contour
#            cv2.drawContours(img_copy, [currentContour], -1, (0,255,0), 2)
#            mask = cv2.bitwise_and(img_copy, mask)
        if currentHierarchy[2] >= 0:
            count_with_holes=count_with_holes+1
            cv2.drawContours(img_copy, [currentContour], -1, (255,0,255), 2)
            mask = cv2.bitwise_and(img_copy, mask)
    print('\nHow many objects have one or more holes?')
    print('Number of objects that have one or more holes:',count_with_holes)        
    return  img_copy       
 
 
img_path=r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment4/input_data/objects.jpg'
img = cv2.imread(img_path)
rgb,gray=getColorSpaces(img)

binary=getBinaryImage(gray)

plt.figure(figsize=(20, 20))

plt.subplot(3,2,1) 
plt.axis('off')
plt.title('Original') 
plt.imshow(gray,cmap='gray')

objects_with_holes=getObjectsWithHoles(binary)
plt.subplot(3,2,2) 
plt.axis('off')
plt.title('Objects with holes') 
plt.imshow(objects_with_holes,cmap='gray')

rect_count, circle_count=getSqaureCircleCount(binary)
print('\nHow many square objects are in the image?')
print('Number of square objects:',rect_count)


square_objects=getSquareObjects(binary)
plt.subplot(3,2,3) 
plt.axis('off')
plt.title('Square Objects') 
plt.imshow(square_objects,cmap='gray')

getObjectsWithHoles(square_objects)

circle_objects=getCircleObjects(binary)
plt.subplot(3,2,4) 
plt.axis('off')
plt.title('Circle Objects') 
plt.imshow(circle_objects,cmap='gray')

getObjectsWithHoles(circle_objects)


