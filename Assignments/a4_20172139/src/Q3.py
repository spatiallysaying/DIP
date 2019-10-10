import numpy as np
import cv2
import os,sys
from matplotlib import pyplot as plt


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


def ifCircular(contour,lowerLimit,UpperLimit):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:
        return False
    circularity = 4*np.pi*(area/(perimeter*perimeter))
    
    if lowerLimit < circularity < UpperLimit:
        return True
    else:
        return False


img_path= 'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment4/input_data/coins.jpg'
img = cv2.imread(img_path)
rgb,gray=getColorSpaces(img)

plt.figure(figsize=(20, 20))


def identifyCircles(gray,min_cnt_area = 320):
    img_copy=img.copy()
    ret,thresh= cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = 1)        
    #If you use  RETR_EXTERNAL flag, it returns only extreme outer flags. All child contours are left behind. 
    _,contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area=cv2.contourArea(contour)    
        if np.ceil(area)/min_cnt_area>1:
            cv2.drawContours(img_copy, [contour], -1, (0,0,255), 2)
            cv2.fillPoly(img_copy, pts =[contour], color=(0,0,255))
        else:
            if np.ceil(area)/min_cnt_area<=1:    
                if ifCircular(contour,0.75,1):
                    cv2.drawContours(img_copy, [contour], -1, (0,255,0), 2)
                    cv2.fillPoly(img_copy, pts =[contour], color=(0,255,0))
    return img_copy


ret,thresh= cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(thresh,kernel,iterations = 1)        
#If you use  RETR_EXTERNAL flag, it returns only extreme outer flags. All child contours are left behind. 
_,contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   
def getOverlappingCircles(gray,min_cnt_area=320):
    mask = np.zeros_like(img)
    for contour in contours:
        area=cv2.contourArea(contour)    
        if np.ceil(area)/min_cnt_area>1:
            cv2.drawContours(mask, [contour], -1, (255,255,255), 2)
            cv2.fillPoly(mask, pts =[contour], color=(0,0,255))
            mask = cv2.bitwise_and(img, mask)
            
    return mask 

def getNonOverlappingCircles(gray,min_cnt_area=320):
          
    mask = np.zeros_like(img)
    for contour in contours:
        area=cv2.contourArea(contour)         
        if np.ceil(area)/min_cnt_area<=1:    
            if ifCircular(contour,0.75,1):
                cv2.drawContours(mask, [contour], -1, (255,255,255), 2)
                cv2.fillPoly(mask, pts =[contour], color=(0,255,0))
                mask = cv2.bitwise_and(img, mask)
    return mask      
            
plt.subplot(3,2,1) 
plt.axis('off')
plt.title('Original') 
plt.imshow(gray,cmap='gray')

circles_all=identifyCircles(gray)

plt.subplot(3,2,2) 
plt.axis('off')
plt.title('Overlapping  & Isolated')
plt.imshow(circles_all,cmap='gray')

circles_overlap=getOverlappingCircles(gray)

plt.subplot(3,2,3) 
plt.axis('off')
plt.title('Overlapping')
plt.imshow(circles_overlap,cmap='gray')

circles_non_overlap=getNonOverlappingCircles(gray)

plt.subplot(3,2,4) 
plt.axis('off')
plt.title('Non-Overlapping')
plt.imshow(circles_non_overlap,cmap='gray')


plt.show()

#cv2.imshow("Image with circles",mask)
#
#k = cv2.waitKey(0)
#if k == 27:
#    cv2.destroyAllWindows()


def getRadius(min_cnt_area=320): 
    circle_area=min_cnt_area #2*np.pi()*r 
    r   =circle_area/(2*np.pi) 
    return r
     

aoi_poly = img.copy()
# Find circles touching image boundary
h, w, _ = img.shape
radius = 10
touching = 0
aoi_corners = np.array(([radius,radius], 
                      [w-radius,radius], 
                      [w-radius, h-radius], 
                      [radius, h-radius]))
cv2.fillPoly(aoi_poly, [aoi_corners], [0,0,0])


plt.axis('off')
plt.title('Circles touching boundary')
plt.imshow(aoi_poly)
plt.show()













