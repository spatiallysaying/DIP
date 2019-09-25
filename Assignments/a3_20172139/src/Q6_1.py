# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:20:58 2019

@author: E442282
"""

import numpy as np
import cv2
import os,sys
from matplotlib import pyplot as plt
import scipy
from PIL import Image
from scipy.signal import convolve2d 




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
    magnitude_spectrum = np.log(1+np.abs(fshift))
#    magnitude_spectrum = 20*np.abs(fshift)
    return magnitude_spectrum
            

def get_ifft_img(img):
    # Take 2d fft
    f = np.fft.ifft2(img)
    # Apply fft shift for visualization
    fshift = np.fft.ifftshift(f)
    # We are only checking the magnitude 
    # We use log as transformation, hence values should be > 1
    magnitude_spectrum = np.log(np.abs(fshift) +1)
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
    


image1= cv2.imread(r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment3/input_data/lena.jpg')
rgb1,gray1=getColorSpaces(image1)
f = cv2.resize(gray1, (256,256))

image2= cv2.imread(r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment3/input_data/bricks.jpg')
rgb2,gray2=getColorSpaces(image2)
h = cv2.resize(gray2, (256,256))


f_Height, f_Width =  f.shape
h_Height, h_Width = h.shape
conv_fh = convolve2d(f,h)
#conv_fh = cv2.filter2D(f,-1,h,borderType = cv2.BORDER_CONSTANT)

ff_conv = np.fft.fft2(conv_fh)
ff_conv = np.absolute(ff_conv)

plt.title("Spatial Convolve")
plt.imshow(np.log(1 + np.abs(ff_conv)), cmap = 'gray')
plt.show()
f_pad = np.pad(f, [(0,h_Height-1), (0,h_Width-1)], mode='constant', constant_values=0)
h_pad = np.pad(h, [(0,f_Height-1), (0,f_Width-1)], mode='constant', constant_values=0)
F = np.fft.fft2(f_pad)
H = np.fft.fft2(h_pad)

FH_product = np.multiply(F,H)
plt.title("Frequency Domain Multiplication")
plt.imshow(np.log(1 + np.abs(FH_product)), cmap = 'gray')
plt.show()
mul_abs = np.absolute(FH_product)

delta = np.sum((mul_abs - ff_conv)**2)
delta = delta/((h_Height + f_Height - 1) * (h_Width + f_Width - 1))
print("Mean Squared error=" ,delta)

















