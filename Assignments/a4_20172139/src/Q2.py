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

def getBinaryImage(gray,threshold):
    ret,thresh= cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    return thresh
 
   

 
img_path=r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment4/input_data/kidney.jpg'
img = cv2.imread(img_path)

 # Remove noise by blurring with a Gaussian filter
img = cv2.GaussianBlur(img, (3, 3), 0)
    
rgb,gray=getColorSpaces(img)



_, binary = cv2.threshold(gray, 131, 255, cv2.THRESH_BINARY)

kernel1=np.array([
	   [0,0, 0, 0, 0, 0, 0],
       [0,0, 0, 0, 0, 0, 0],
       [0,0, 1, 1, 1, 0, 0],
       [0,0, 1, 1, 1, 0, 0],
       [0,0, 1, 1, 1, 0, 0],
       [0,0, 1, 1, 1, 0, 0],
	   [0,0, 0, 0, 0, 0, 0],
	   [0,0, 0, 0, 0, 0, 0],
	   
	   ], dtype=np.uint8)
    
    
#binary = cv2.morphologyEx(binary, cv2.MORPH_HITMISS, kernel1)    


#closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


#erosion = cv2.erode(binary,kernel,iterations = 1)
#dilation = cv2.dilate(erosion,kernel,iterations = 1)

#opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) 

se_circular=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))



opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se_circular)
#
#kernel = np.ones((5,5),np.uint8)
#closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#erosion = cv2.erode(binary,kernel,iterations = 1)
#closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((3,3),np.uint8)
dilation = cv2.dilate(opening,kernel,iterations = 1)


plt.figure(figsize=(20, 20))

plt.subplot(3,2,1) 
plt.axis('off')
plt.title('Original') 
plt.imshow(gray,cmap='gray')

plt.subplot(3,2,2) 
plt.axis('off')
plt.title('Thresholded') 
plt.imshow(binary,cmap='gray')

plt.subplot(3,2,3) 
plt.axis('off')
plt.title('Opening') 
plt.imshow(opening,cmap='gray')

plt.subplot(3,2,4) 
plt.axis('off')
plt.title('Dilation') 
plt.imshow(dilation,cmap='gray')


ret, labels = cv2.connectedComponents(dilation)

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('labeled.png', labeled_img)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()

imshow_components(labels)


#
#def nothing(x):
#    pass
#
#cv2.namedWindow("Image")
#cv2.createTrackbar("Threshold value", "Image", 128, 255, nothing)
#while True:
#    value_threshold = cv2.getTrackbarPos("Threshold value", "Image")
#    _, threshold_binary = cv2.threshold(gray, value_threshold, 255, cv2.THRESH_BINARY)
#
#    cv2.imshow("Image", gray)
#    cv2.imshow("th binary", threshold_binary)
#
#    key = cv2.waitKey(100)
#    if key == 27:
#        break
#cv2.destroyAllWindows()

