# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:20:58 2019

@author: E442282
"""

import numpy as np
import cv2
import os,sys
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided


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
  red, green, blue= cv2.split(image)
  
  return red, green, blue
 
    
def getNormalizedImage(image):
    norm_image=image.copy()
    norm_image = np.maximum(norm_image, np.zeros(norm_image.shape))
    norm_image = np.minimum(norm_image, 255 * np.ones(norm_image.shape))
    norm_image = norm_image.round().astype(np.uint8)
    
    return norm_image
 

def getFFT2D(gray):
    f = np.fft.fft2(gray)
    #DC component is at top-left corner
    #Therefore shift it to right&down so that DC is at center
    fshift = np.fft.fftshift(f)
    
#    %In Fourier transforms, high peaks are so high they 
#%hide details. Reduce contrast with the log function
    magnitude_spectrum = 20*np.log(1+np.abs(fshift))
#    magnitude_spectrum = 20*np.abs(fshift)
    return magnitude_spectrum
            
def getRotatedImage(gray,angle):
    h,w=getImageDimnesion(gray)
    center = (w / 2, h / 2)    
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(gray, M, (w, h))
    
    return rotated


def getTranslatedImage(gray,xShift,yShift):
    h,w=getImageDimnesion(gray)
    
    M = np.float32([[1,0,xShift],[0,1,yShift]])
    translated = cv2.warpAffine(gray,M,(w,h))

    
    return translated


'''
Rotating f(x, y) by an angle 'Theta' rotates F(u, v) by the same
angle. Conversely, rotating F(u, v) rotates f(x, y) by the same angle

Rotating f(x,y) by θ rotates F(u,v) by θ
'''


img = cv2.imread(r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment3/input_data/rectangle.jpg')
plt.figure(figsize=(12, 12))
rgb,gray=getColorSpaces(img)

gray_rotated=getRotatedImage(gray,-45)
gray_translated=getTranslatedImage(gray,50,50)


plt.subplot(4,2,1) 
plt.axis('off')
plt.title('Original') 
plt.imshow(gray,cmap='gray')
 
plt.subplot(4,2,2) 
plt.axis('off')
plt.title('Magnitude ')
plt.imshow(getFFT2D(gray),cmap='gray')

plt.subplot(4,2,3) 
plt.axis('off')
plt.title('Rotated') 
plt.imshow(gray_rotated,cmap='gray')
  
plt.subplot(4,2,4) 
plt.axis('off')
plt.title('Magnitude of Rotated ')
plt.imshow(getFFT2D(gray_rotated),cmap='gray')


plt.subplot(4,2,5) 
plt.axis('off')
plt.title('Translated') 
plt.imshow(gray_translated,cmap='gray')
  
plt.subplot(4,2,6) 
plt.axis('off')
plt.title('Magnitude of Translated ')
plt.imshow(getFFT2D(gray_translated),cmap='gray')


gray_translated=getTranslatedImage(gray,-50,-100)

plt.subplot(4,2,7) 
plt.axis('off')
plt.title('Translated') 
plt.imshow(gray_translated,cmap='gray')
  

plt.subplot(4,2,8) 
plt.axis('off')
plt.title('Magnitude of Translated ')
plt.imshow(getFFT2D(gray_translated),cmap='gray')
